# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import statistics
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from volta.config import BertConfig, M3PConfig
from volta.encoders import BertForVLPreTraining, BertForVLTasks, M3PForVLTasks
from volta.task_utils import LoadDatasetTest

import wandb
import analysis_tools as tools


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", 
                        default="/root/autodl-tmp/volta/checkpoints/mmdata/ctrl_vilbert/RetrievalMMdata_ctrl_vilbert_base/pytorch_model_9.bin", 
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # parser.add_argument("--from_pretrained", default="/root/autodl-tmp/volta/checkpoints/mmdata/ctrl_vilbert/RetrievalMMdata_ctrl_vilbert_base/pytorch_model_9.bin", type=str,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    # parser.add_argument("--from_pretrained", default="/root/autodl-tmp/volta/exmaple_xinyi_foil/checkpoints/ctrl_vilbert_base/FOIL_annotations_all/pytorch_model_best.bin", type=str,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="/root/autodl-tmp/volta/config/ctrl_vilbert_base.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--is_m3p", action='store_true', default=False,
                        help="Use M3P.")
    # Output
    parser.add_argument("--output_dir", default="/root/autodl-tmp/volta/exmaple_xinyi_bla_train/results/check",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    # parser.add_argument("--tasks_config_file", default="/root/autodl-tmp/volta/exmaple_xinyi_bla_train/task_configs/ctrl_active_tasks.yml", type=str,
    #                     help="The config file which specified the tasks details.")
    # parser.add_argument("--eval_num_set_size", default=4, type=int,
    #                     help="The number of sentences in one caption set of validation set")
    parser.add_argument("--tasks_config_file", default="/root/autodl-tmp/volta/exmaple_xinyi_bla/ctrl_coordination_tasks.yml", type=str,
                        help="The config file which sp")
    parser.add_argument("--eval_num_set_size", default=4, type=int,
                        help="The number of sentences in one caption set of validation set")
    parser.add_argument("--task", default="8", type=str,
                        help="training task number")
    # parser.add_argument("--val_annotations_jsonpath", 
    #                     default="/root/autodl-tmp/datasets/BLA/annotations/active_passive_captions_gruen_strict_ann.jsonl", 
    #                     type=str)
    parser.add_argument("--test_annotations_jsonpath", default="", type=str)
    # parser.add_argument("--val_features_lmdbpath", 
    #                     default="/root/autodl-tmp/datasets/BLA/imgfeats/volta/genome_feat.lmdb", 
    #                     type=str)
    parser.add_argument("--test_features_lmdbpath", default="", type=str)
    parser.add_argument("--num_subiters", default=1, type=int)
    parser.add_argument("--caps_per_image", default=5, type=int,
                        help="Num captions per image")
    # Evaluation
    parser.add_argument("--split", default="test", type=str,
                        help="which split to use.")
    parser.add_argument("--zero_shot", action="store_true",
                        help="Zero-shot evaluation.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="batch size.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default= -1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--num_val_workers", type=int, default=10)
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    parser.add_argument("--use_chunk", default=0, type=float,
                        help="whether use chunck for parallel training.")

    return parser.parse_args()


def main():
    args = parse_args()
    args.zero_shot = True
    model_name = args.from_pretrained.split('/')[-2]
    task_name = args.tasks_config_file.split('/')[-1].split('.')[0]
    
    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # n_gpu = torch.cuda.device_count()
        # prevent acc caculation errors
        n_gpu = 1
        
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    
    default_gpu = False
    
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    if args.is_m3p:
        config = M3PConfig.from_json_file(args.config_file)
    else:
        config = BertConfig.from_json_file(args.config_file)

    # Load task config
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id

    # Output dirs
    if "/" in args.from_pretrained:
        timeStamp = args.from_pretrained.split("/")[1]
    else:
        timeStamp = args.from_pretrained
    # savePath = os.path.join(args.output_dir,savePath)
    savePath = os.path.join(args.output_dir)
    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    batch_size, task2num_iters, dset_val, dl_val = LoadDatasetTest(args, config, task_cfg, args.task)
    
    # Model
    if args.zero_shot:
        config.visual_target_weights = {}  # [0, 0, 0, 0, 0, 0, 0]
        model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config)
    else:
        if args.is_m3p:
            model = M3PForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])
        else:
            model = BertForVLTasks.from_pretrained(args.from_pretrained, config=config, task_cfg=task_cfg, task_ids=[task])

    # Move to GPU(s)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, deay_allreduce=True)
    elif n_gpu > 1:
        try:
            model = nn.DataParallel(model)
        except:
            raise ValueError("Please run with a single GPU")

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", task2num_iters)
        print("  Batch size: ", batch_size)

    # Evaluate
    model.eval()
    results = []
    others = []
    
    correct_cnt = 0
    for i, batch in tqdm(enumerate(dl_val), total=task2num_iters[task]):
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, question, input_mask, \
            segment_ids, target, caption_idx, image_id = batch
        
        question = question.squeeze(0)
        segment_ids = segment_ids.squeeze(0)
        input_mask = input_mask.squeeze(0)


        features = features.repeat(question.size(0), 1, 1)
        spatials = spatials.repeat(question.size(0), 1, 1)
        image_mask = image_mask.repeat(question.size(0), 1)
        
        # question = question.repeat(features.size(0), 1)
        # segment_ids = segment_ids.repeat(features.size(0), 1)
        # input_mask = input_mask.repeat(features.size(0), 1)

        target = target.view(-1).float().cpu().numpy()
        vil_logits = np.zeros(len(target))
        
        for j in range(len(target)):
            question_j = question[j].unsqueeze(0)
            features_j = features[j].unsqueeze(0)
            spatials_j = spatials[j].unsqueeze(0)
            segment_ids_j = segment_ids[j].unsqueeze(0)
            input_mask_j = input_mask[j].unsqueeze(0)
            image_mask_j = image_mask[j].unsqueeze(0)
            
            with torch.no_grad():
                if args.zero_shot:
                    # vil_logit is seq_relationship_score, shape [1,2]
                    # seq_relationship_score = bi_seq_relationship(pooled_output)
                    # pooled_output is the multiply results, relationship between two modality 
                    # bi_seq_relationship = nn.Linear(config.pooler_size, config.itm_dim)
                    _, _, vil_logit, _, _, _ = model(question_j, features_j, spatials_j,
                                                     segment_ids_j, input_mask_j, image_mask_j)
                    # vil_logits_orginal[j] = vil_logit[:,0]
                    # print(i, j, question_j.shape, features_j.shape, spatials_j.shape, segment_ids_j.shape, input_mask_j.shape, image_mask_j.shape)
                    # print(i, image_id, j, vil_logit.view(-1).cpu().numpy())
                    vil_logit = torch.softmax(vil_logit, dim=1)[:, 0]
                
                else:
                    # question: input_text, features:input_img, spatials:image_loc
                    vil_logit, _, _, _ = model(question_j, features_j, spatials_j,
                                               task, segment_ids_j, input_mask_j,
                                               image_mask_j)
             
                vil_logit = vil_logit.view(-1).cpu().numpy()
                vil_logits[j] = vil_logit
        
        sorted_idx = vil_logits.argsort()
        rank = list(range(len(sorted_idx)))
        for idx, v in enumerate(sorted_idx):
            rank[v] = len(rank) - idx

        # results.append({"image_id": image_id.item(), "rank": rank, \
        #     "scores": vil_logits.tolist()})
        results.append({"image_id": image_id.item(), "rank": rank, \
            "scores": vil_logits})
        
        if len(target) <= 2:
            correct_cnt += 1 if rank[0] == 1 else 0
            if rank[0] != 1:
                others.append(f"{i}: { vil_logits}\n")
        #     # results.append({"image_idx": i, "rank": rank.tolist(), "scores": vil_logit.tolist()})
        #     print(correct_cnt/ (i + 1))
        
    if "ctrl_" in model_name and args.test_annotations_jsonpath != "":
        dataset_names = args.test_annotations_jsonpath.split("/")
        dataset_task_type = dataset_names[-2] + "_" + dataset_names[-1][:-10]
        json_path = os.path.join(savePath, dataset_task_type + "_pretrained" )
    else:
        json_path = os.path.join(savePath, model_name)
    

    if len(target) > 2:
        result_ranks = [r["rank"] for r in results]
        rank_statistics = tools.get_rank_statistics(result_ranks)
        # cls_statistics = tools.get_bi_cls_statistics(results, 0.5)
        print(rank_statistics['rank_acc'])
        # score_statistics['rank_acc'] = statistics['sent_acc']
        # score_statistics["total"] = statistics['total']
        # score_statistics["model"] = model_name
        # score_statistics["task"] = task_name
        # json.dump(score_statistics, open(json_path + "_result.json", "w"), indent=2)
    else:
        result_ranks = [r["rank"] for r in results]
        rank_statistics = tools.get_rank_statistics(result_ranks)
        print("rank acc", rank_statistics['rank_acc'])
        print(correct_cnt / len(dl_val))
    
    # json.dump(others, open(json_path + "_model_outputs.json", "w"), indent=2)
    # print(others)
    print("Results saved at", json_path + "_result.json")
    print("finish")


if __name__ == "__main__":
    main()
