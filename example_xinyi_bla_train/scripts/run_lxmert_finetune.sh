#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=bla_lxmert_finetune
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=bla_lxmert_finetune_cls.out

module load 2021
module load  Anaconda3/2021.05
source activate volta

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


cd $HOME/volta-bla
echo "lxmert Results Train"

# TASK_TYPE=ctrl_active_tasks
SPLIT_TYPE_RANDOM=finetune_random
# SPLIT_TYPE_TRAIN_GOLDEN=finetune_train_golden_large
# SPLIT_TYPE_TEST_GOLDEN=finetune_test_golden

ANNOTATIONS_PATH=/home/xchen/datasets/BLA_finetune/annotations
FEATURES_PATH=/home/xchen/datasets/BLA_finetune/imgfeats/volta-bla
TRAIN_BATCH_SIZE=16

LXMERT_MODEL=ctrl_lxmert
LXMERT_MODEL_CONFIG=ctrl_lxmert
LXMERT_PRETRAINED=/home/xchen/volta-bla/checkpoints/mmdata/${LXMERT_MODEL}/RetrievalMMdata_${LXMERT_MODEL_CONFIG}/pytorch_model_9.bin
VILBERT_OUTPUT_DIR=/home/xchen/volta-bla/example_xinyi_bla_train/results/${LXMERT_MODEL}/active
FREEZE_BEFORE_LAYER=-1

cd $HOME/volta-bla
echo "Vilbert Results Train"

python train_concap_modify.py \
    --tasks_config_file example_xinyi_bla_train/task_configs/ctrl_active_tasks.yml\
    --config_file config/${LXMERT_MODEL_CONFIG}.json --from_pretrained ${LXMERT_PRETRAINED}\
    --train_batch_size ${TRAIN_BATCH_SIZE} --eval_batch_size 1 --num_train_epochs 10 \
    --freeze_before_layer ${FREEZE_BEFORE_LAYER} --validation_metrics cls_acc


python train_concap_modify.py \
    --tasks_config_file example_xinyi_bla_train/task_configs/ctrl_coord_tasks.yml\
    --config_file config/${LXMERT_MODEL_CONFIG}.json --from_pretrained ${LXMERT_PRETRAINED}\
    --train_batch_size ${TRAIN_BATCH_SIZE} --eval_batch_size 1 --num_train_epochs 10 \
    --freeze_before_layer ${FREEZE_BEFORE_LAYER} --validation_metrics cls_acc

python train_concap_modify.py \
    --tasks_config_file example_xinyi_bla_train/task_configs/ctrl_rc_tasks.yml\
    --config_file config/${LXMERT_MODEL_CONFIG}.json --from_pretrained ${LXMERT_PRETRAINED}\
    --train_batch_size ${TRAIN_BATCH_SIZE} --eval_batch_size 1 --num_train_epochs 10 \
    --freeze_before_layer ${FREEZE_BEFORE_LAYER} --validation_metrics cls_acc

conda deactivate
