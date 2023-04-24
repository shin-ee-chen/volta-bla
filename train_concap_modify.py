# Copyright (c) Facebook, Inc. and its affiliates.

# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import os
import sys
import json
import random
import logging
import argparse
from io import open
import datetime
# import tensorpack.dataflow as td
import yaml
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from transformers import AutoTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from volta.config import BertConfig
from volta.encoders import BertForVLPreTraining
# from volta.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal, ConceptCapLoaderTest
from volta.train_utils import freeze_layers, tbLogger, summary_parameters, save, resume
from volta.task_utils import LoadDatasetEval, LoadDatasetTrain, LoadDatasetTest

import analysis_tools as tools

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    dataset_type = "finetune_random/active_passive"
    
    # Data
    parser.add_argument("--tasks_config_file", default="/home/xchen/volta-bla/exmaple_xinyi_bla_train/task_configs/ctrl_active_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    # parser.add_argument("--tasks_config_file", default="/home/xchen/volta-bla/exmaple_xinyi_bla_train/task_configs/ctrl_active_tasks.yml", type=str,
    #                     help="The config file which specified the tasks details.")
    parser.add_argument("--train_num_set_size", default=2, type=int,
                        help="The number of sentences in one caption set of training set")
    parser.add_argument("--eval_num_set_size", default=4, type=int,
                        help="The number of sentences in one caption set of validation set")
    
    parser.add_argument("--task", default="8", type=str,
                        help="training task number")
    
    parser.add_argument("--train_annotations_jsonpath", default="", type=str)
    parser.add_argument("--train_features_lmdbpath", default="", type=str)
    
    parser.add_argument("--val_annotations_jsonpath", default="", type=str)
    parser.add_argument("--val_features_lmdbpath", default="", type=str)
    
    parser.add_argument("--test_annotations_jsonpath", default="", type=str)
    parser.add_argument("--test_features_lmdbpath", default="", type=str)
    
    parser.add_argument("--train_split", default="", type=str)
    parser.add_argument("--val_split", default="test", type=str)
    parser.add_argument("--split", default="test", type=str,
                        help="which split to use.")
    parser.add_argument("--num_val_workers", type=int, default=10)
    parser.add_argument("--num_subiters", default=1, type=int)
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    
    # Model
    parser.add_argument("--from_pretrained", default="/home/xchen/volta-bla/checkpoints/mmdata/ctrl_vilbert/RetrievalMMdata_ctrl_vilbert_base/pytorch_model_9.bin", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", type=str, default="/home/xchen/volta-bla/config/ctrl_vilbert_base.json",
                        help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    parser.add_argument("--save_model", action='store_true',
                        help="Whether to save model")

    # Output
    parser.add_argument("--output_dir", default="/home/xchen/volta-bla/exmaple_xinyi_bla_train/checkpoints", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="/home/xchen/volta-bla/exmaple_xinyi_bla_train/logs", type=str,
                        help="The logging directory where the training logs will be written.")
    # parser.add_argument("--output_dir", default="/home/xchen/volta-bla/exmaple_xinyi_foil/checkpoints", type=str,
    #                     help="The output directory where the model checkpoints will be written.")
    # parser.add_argument("--logdir", default="/home/xchen/volta-bla/exmaple_xinyi_foil/logs", type=str,
    #                     help="The logging directory where the training logs will be written.")
    
    # Text
    parser.add_argument("--max_seq_length", default=20, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # Training
    parser.add_argument("--freeze_before_layer", default=-1, type=int,
                        help="Finetune the entire model or the last") 
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for training.")
    
    
    parser.add_argument("--learning_rate", default=4e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    # Scheduler
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--distributed", action="store_true",
                        help="whether use chunck for parallel training.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    
    # Objective
    parser.add_argument("--objective", default=0, type=int,
                        help="Which objective to use \n"
                             "0: with ITM loss, \n"
                             "1: with ITM loss; for the not aligned pair, no masking objective, \n"
                             "2: without ITM loss, do not sample negative pair.")
    # Optimizer
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.98), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")

    return parser.parse_args()



def evaluate_model(dl_val, task2num_iters, task, model, device):
    print("Start model evaluation")
    torch.set_grad_enabled(False)
    model.eval()
    rank_predictions = []
    correct_cnt = 0
    
    sum_pair_match_loss, sum_masked_loss_t, sum_masked_loss_v = 0, 0, 0
    loss_steps = 0
    vil_logit_scores = []
    targets = []
    rank_logits = []
    
    for step, batch in tqdm(enumerate(dl_val), total=task2num_iters[task]):
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, question, input_mask, \
            segment_ids, target, caption_idx, image_id = batch
        
        question = question.squeeze(0)
        segment_ids = segment_ids.squeeze(0)
        input_mask = input_mask.squeeze(0)

        features = features.repeat(question.size(0), 1, 1)
        spatials = spatials.repeat(question.size(0), 1, 1)
        image_mask = image_mask.repeat(question.size(0), 1)
        
        # target = target.view(-1).float().cpu().numpy()
        target = target.view(-1).unsqueeze(1)
        vil_logits = []
        # vil_logits = torch.zeros(target.shape[0])

        for j in range(len(target)):
            target_j = target[j].unsqueeze(0)
            question_j = question[j].unsqueeze(0)
            features_j = features[j].unsqueeze(0)
            spatials_j = spatials[j].unsqueeze(0)
            segment_ids_j = segment_ids[j].unsqueeze(0)
            input_mask_j = input_mask[j].unsqueeze(0)
            image_mask_j = image_mask[j].unsqueeze(0)
            
            with torch.no_grad():
                _, _, vil_logit, _, _, _ = model(question_j, features_j, spatials_j,
                                                 segment_ids_j, input_mask_j, image_mask_j)
                
                
                _, _, pair_match_loss = model(question_j, features_j, spatials_j,
                                              segment_ids_j, input_mask_j, image_mask_j, 
                                              next_sentence_label = 1 - target_j)
                masked_loss_t, masked_loss_v = 0, 0
                vil_logit_scores.append(vil_logit)
                targets.append(target_j)
                
                sum_pair_match_loss += pair_match_loss
                loss_steps += 1

                vil_logit = torch.softmax(vil_logit, dim=1)[:, 0]
              
                # vil_logit = vil_logit.view(-1).cpu().numpy()
                vil_logits.append(vil_logit.view(-1))
        
            
        rank_logits.append(vil_logits)
        
        # sorted_idx = vil_logits.argsort()
        # rank = list(range(len(sorted_idx)))
        # for idx, v in enumerate(sorted_idx):
        #     rank[v] = len(rank) - idx

        # rank_predictions.append({"image_id": image_id.item(), "rank": rank, \
        #     "scores": vil_logits.tolist()})
        
        # if len(target) <= 2:
        #     correct_cnt += 1 if rank[0] == 1 else 0
    rank_logits = torch.tensor(rank_logits)
    rank_predictions = len(target) - torch.argsort(torch.argsort(rank_logits , dim = 1), dim = 1).cpu()
    targets = torch.tensor(targets)
    
    if len(target) > 2:
        statistics = tools.get_rank_statistics(rank_predictions)
        bi_cls_results = tools.get_bi_cls_statistics(torch.cat(vil_logit_scores, 0).cpu(), 
                                                     targets)
        statistics.update(bi_cls_results)
        # score_statistics["model"] = model_name
        # score_statistics["task"] = task_name
        # json.dump(score_statistics, open(json_path + "_result.json", "w"), indent=2)
        # print("Evaluation accuracy:", statistics['sent_acc'], score_statistics)
    else:
        statistics = {}
        correct_cnt = ((rank_predictions - 1) == (1 - targets.reshape(-1, 2))).sum() / 2
        statistics['rank_acc'] = np.round(correct_cnt / len(rank_predictions), 2)
        statistics["total"] = len(rank_predictions)
        bi_cls_results = tools.get_bi_cls_statistics(torch.cat(vil_logit_scores, 0).cpu(), 
                                                     torch.tensor(targets))
        statistics.update(bi_cls_results)
        # print("Evaluation accuracy:", correct_cnt / len(dl_val))
    
    return statistics, sum_pair_match_loss/loss_steps, \
            sum_masked_loss_t, sum_masked_loss_v


def main():
    args = parse_args()
    # if args.tasks_config_file[-1] == '/':
    #     args.tasks_config_file = args.tasks_config_file[:-1]
    dataset_type = args.tasks_config_file.split('/')[-1].split(".")[0]
    freeze_type = "last" if args.freeze_before_layer > 0 else "all"
    model_type = args.config_file.split("/")[-1].split(".")[0]
    # task name, split type, freeze layer
    hyperparas = str(args.learning_rate) + "_" \
                 + str(args.train_batch_size)
    model_name = args.from_pretrained.split('/')[-3]
    model_name = dataset_type + "_" + freeze_type + "_" + \
                    hyperparas+ "_" + str(args.seed) + "_" + model_name
    
    json_path = os.path.join(args.output_dir, model_name)
    
    args.logdir = os.path.join(args.logdir, model_name)
    # args.save_model = True
    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Init distributed backend for sychronizing nodes/GPUs
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config
    config = BertConfig.from_json_file(args.config_file)

    # Output dirs
    save_path = os.path.join(args.output_dir, model_type, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if default_gpu:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    cache = 5000
    # cache = 10
    args.train_batch_size = args.train_batch_size // args.grad_acc_steps
    if dist.is_available() and args.local_rank != -1:
        num_replicas = dist.get_world_size()
        args.train_batch_size = args.train_batch_size // num_replicas
        args.num_workers = args.num_workers // num_replicas
        cache = cache // num_replicas

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Datasets
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)
   
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    task_names = ["BLA"]
    task_ids = [task]
    args.cache = 5000
    
    train_batch_size, task2num_iters, dset_train, dl_train = LoadDatasetTrain(args, config, task_cfg, args.task, True, True)
    test_batch_size, task2num_iters, dset_test, dl_test = LoadDatasetTest(args, config, task_cfg, args.task, False)
    val_batch_size, task2num_iters, dset_val, dl_val = LoadDatasetEval(args, config, task_cfg, args.task, False)
   
    # Logging
    logdir = args.logdir
    if default_gpu:
        tb_logger = tbLogger(logdir, save_path, task_names, task_ids, task2num_iters, args.grad_acc_steps)
    else:
        tb_logger = None
        
    
    # Model
    if args.from_pretrained:
        type_vocab_size = config.type_vocab_size
        config.type_vocab_size = 2
        # model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config,
        #                                              default_gpu=default_gpu, from_hf=True)
        model = BertForVLPreTraining.from_pretrained(args.from_pretrained, config=config, 
                                                     default_gpu=default_gpu)
        
        # Resize type embeddings
        model.bert.embeddings.token_type_embeddings = \
            model._get_resized_embeddings(model.bert.embeddings.token_type_embeddings, type_vocab_size)
        config.type_vocab_size = type_vocab_size
    else:
        model = BertForVLPreTraining(config)
    
    # Optimization details
    freeze_layers(model, args.freeze_before_layer)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # bert_weight_name = json.load(open("config/" + args.from_pretrained + "_weight_name.json", "r"))
    # bert_weight_name = json.load(open("/home/xchen/volta-bla/config/bert-base-uncased_weight_name.json", "r"))
    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                # if key[12:] in bert_weight_name:
                #     lr = args.learning_rate * 0.1
                # else:
                lr = args.learning_rate
                
                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=args.adam_betas)
    num_train_optimization_steps = int(
        len(dset_val)
        / args.train_batch_size
        / args.grad_acc_steps
    ) * args.num_train_epochs
    warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optimization_steps
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    # Resume training
    start_iter_id, global_step, start_epoch, tb_logger, _ = \
        resume(args.resume_file, model, optimizer, scheduler, tb_logger)

    # Move to GPU(s)
    model.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Print summary
    if default_gpu:
        summary_parameters(model, logger)
        logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", train_dataset.num_dataset)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
    
    # Evaluation before training (Validation)
    numBatches = len(dl_test)
    results, pair_match_loss, masked_loss_t, masked_loss_v = evaluate_model(dl_test, 
                                                                            task2num_iters, 
                                                                            task, 
                                                                            model, 
                                                                            device)
    output_records = []
    print(f"Test Set Validation, epoch = {-1}, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}")
    output_records.append(f"Test Set Validation, epoch = {-1}, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}, {results}")
    # Evaluation before training (Validation)
    numBatches = len(dl_val)
    results, pair_match_loss, masked_loss_t, masked_loss_v = evaluate_model(dl_val, 
                                                                            task2num_iters, 
                                                                            task, 
                                                                            model, 
                                                                            device)
    
    print(f"epoch = {-1}, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}")
    output_records.append(f"epoch = {-1}, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}")
    best_rank_model = {"epoch": -1, "model": model, "result": results['rank_acc']}
    best_loss_model = {"epoch": -1, "model": model, "result": pair_match_loss}
    
    # Train
    torch.set_grad_enabled(True)
    for epoch_id in range(start_epoch, int(args.num_train_epochs)):
        model.train()
        train_pair_loss = []
        for step, batch in tqdm(enumerate(dl_train), total=task2num_iters[task]):
            iter_id = start_iter_id + step + (epoch_id * len(dl_train))
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            features, spatials, image_mask, question, input_mask, \
                segment_ids, target, caption_idx, image_id = batch
        
            features = features.repeat(1, question.size(1), 1)
            spatials = spatials.repeat(1, question.size(1), 1)
            image_mask = image_mask.repeat(1, question.size(1))
        
            features = features.reshape(question.size(0) * question.size(1), 
                                        int(features.size(1) / question.size(1)), 
                                        -1)
            spatials = spatials.reshape(question.size(0) * question.size(1), 
                                        int(spatials.size(1) / question.size(1)), 
                                        -1)

            image_mask = image_mask.reshape(question.size(0) * question.size(1), 
                                            int(image_mask.size(1) / question.size(1)))
            
            #torch.Size([216, , 38])->[32, 38]
            segment_ids = segment_ids.reshape(question.size(0) * question.size(1), -1)
            input_mask = input_mask.reshape(question.size(0) * question.size(1), -1)
            question = question.reshape(question.size(0) * question.size(1), -1)

            target = target.view(-1)
        
            # Batch =1, [1, 38], [1, 37,2048], [1, 37, 5],| [1, 38], [1, 38], [1, 37]
            # _, _, vil_logit, _, _, _ = model(question, features, spatials,
            #                              segment_ids, input_mask, image_mask)
            _, _, pair_match_loss = model(question, features, spatials,
                                          segment_ids, input_mask, image_mask,
                                          next_sentence_label = 1 - target)
            
            masked_loss_t, masked_loss_v = 0, 0
            
            loss = pair_match_loss
            
            if n_gpu > 1:
                loss = loss.mean()
                pair_match_loss = pair_match_loss.mean()

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            
            # train_pair_loss.append(pair_match_loss)
            
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                # Clip gradient
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if default_gpu:
                    tb_logger.step_train_CC(epoch_id, iter_id,
                                            float(masked_loss_t), float(masked_loss_v), float(pair_match_loss),
                                            optimizer.param_groups[0]["lr"], task, "train")

            if (step % (20 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                tb_logger.showLossTrainCC()

        # Do the evaluation
        numBatches = len(dl_val)
        results, pair_match_loss, masked_loss_t, masked_loss_v = evaluate_model(dl_val, 
                                                                                task2num_iters, 
                                                                                task, 
                                                                                model, 
                                                                                device)
        print(f"epoch = {epoch_id}, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}")
        output_records.append(f"epoch = {epoch_id}, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}")
            
        # Save best model
        if results['rank_acc'] > best_rank_model["result"]:
            best_rank_model["model"] = copy.deepcopy(model)
            best_rank_model["result"] = results['rank_acc']
            best_rank_model["epoch"] = epoch_id
        
            if args.save_model:
                save(save_path, logger, epoch_id, model, optimizer, scheduler, global_step, tb_logger, default_gpu, best_rank_model["result"], is_best=True)
            # save(save_path, logger, -1, model, optimizer, scheduler, global_step, tb_logger, default_gpu, -1)
        
        if pair_match_loss < best_loss_model["result"]:
            best_loss_model["model"] = copy.deepcopy(model)
            best_loss_model["result"] = pair_match_loss
            best_loss_model["epoch"] = epoch_id
        
        
        if default_gpu:
                # step_val_CC(self, iter_id, masked_loss_t, masked_loss_v, next_sentence_loss, task_id, batch_size, split)
                tb_logger.step_val_BLA(iter_id, float(results['rank_acc']), float(results['cls_acc']), 
                                      float(pair_match_loss), task, args.eval_batch_size, "val")
                
                sys.stdout.write("%d / %d \r" % (step, numBatches))
                sys.stdout.flush()
        
        torch.set_grad_enabled(True)
   
   
    print(model_name, dataset_type)
    print("Finetune model:", args.freeze_before_layer)
    
    # Evaluation after training
    numBatches = len(dl_test)
    
    # Best rank_acc model
    print(f"Best rank_acc model is {best_rank_model['epoch']}, best validation result is {best_rank_model['result']}")
    output_records.append(f"Best model is {best_rank_model['epoch']}, best validation result is {best_rank_model['result']}")
    
    results, pair_match_loss, masked_loss_t, masked_loss_v = evaluate_model(dl_test, 
                                                                            task2num_iters, 
                                                                            task, 
                                                                            best_rank_model['model'], 
                                                                            device)
    
    print(f"Test Set Validation of best rank_acc model, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}")
    output_records.append(f"Test Set Validation of best model, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}, {results}")
    
    # Best loss model
    print(f"Best loss model is {best_loss_model['epoch']}, best validation result is {best_loss_model['result']}")
    output_records.append(f"Best loss model is {best_loss_model['epoch']}, best validation result is {best_loss_model['result']}")
    
    results, pair_match_loss, masked_loss_t, masked_loss_v = evaluate_model(dl_test, 
                                                                            task2num_iters, 
                                                                            task, 
                                                                            best_loss_model["model"], 
                                                                            device)
    
    print(f"Test Set Validation of best loss model, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}")
    output_records.append(f"Test Set Validation of best loss model, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}, {results}")
    
    # Evaluation after training
    numBatches = len(dl_test)
    print(f"Last model test")
    
    results, pair_match_loss, masked_loss_t, masked_loss_v = evaluate_model(dl_test, 
                                                                            task2num_iters, 
                                                                            task, 
                                                                            model, 
                                                                            device)
    
    print(f"Test Set Validation of last model, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}")
    output_records.append(f"Test Set Validation of last model, rank_acc = {results['rank_acc']}, cls_acc = {results['cls_acc']}, loss = {pair_match_loss}, {results}")
    
    if default_gpu:
        tb_logger.txt_close()
    
    json.dump(output_records, open(json_path + "_result.json", "w"), indent=2)
    

if __name__ == "__main__":
    main()
    print("Finish")
