#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=bla_ap_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=bla_ap_eval_finetune.out

TASK=8
TASKS_CONFIG=ctrl_ap_tasks_test

VILBERT_MODEL=ctrl_vilbert
VILBERT_MODEL_CONFIG=ctrl_vilbert_base
VILBERT_PRETRAINED=/home/xchen/volta-bla/example_xinyi_bla_train/checkpoints/ctrl_vilbert_base/ctrl_active_tasks_all_4e-05_32_42_ctrl_vilbert/pytorch_model_best.bin
VILBERT_OUTPUT_DIR=/home/xchen/volta-bla/example_xinyi_bla_eval/results/${VILBERT_MODEL}/ap


# LXMERT_MODEL=ctrl_lxmert
# LXMERT_MODEL_CONFIG=ctrl_lxmert
# LXMERT_PRETRAINED=/home/xchen/volta-bla/checkpoints/mmdata/${LXMERT_MODEL}/RetrievalMMdata_${LXMERT_MODEL_CONFIG}/pytorch_model_9.bin
# LXMERT_OUTPUT_DIR=/home/xchen/volta-bla/example_xinyi_bla_eval/results/${LXMERT_MODEL}/rc

module load 2021
module load  Anaconda3/2021.05
source activate volta

rm -r /home/xchen/datasets/BLA/annotations/cache
cd $HOME/volta-bla
echo "VILBERT results"

python eval_retrieval_modified.py \
	--config_file config/${VILBERT_MODEL_CONFIG}.json --from_pretrained ${VILBERT_PRETRAINED} \
	--tasks_config_file example_xinyi_bla_eval/task_configs/${TASKS_CONFIG}.yml --task $TASK \
	--split test --eval_batch_size 1 \
	--output_dir ${VILBERT_OUTPUT_DIR} --zero_shot

# echo "LXMERT results"
# python eval_retrieval_modified.py \
# 	--config_file config/${LXMERT_MODEL_CONFIG}.json --from_pretrained ${LXMERT_PRETRAINED} \
# 	--tasks_config_file example_xinyi_bla_eval/task_configs/${TASKS_CONFIG}.yml --task $TASK --split test --eval_batch_size 1 \
# 	--output_dir ${LXMERT_OUTPUT_DIR} --zero_shot

conda deactivate
