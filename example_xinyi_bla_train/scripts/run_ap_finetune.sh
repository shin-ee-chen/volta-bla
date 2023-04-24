#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=bla_rc_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=bla_rc_eval.out

module load 2021
module load  Anaconda3/2021.05
source activate volta


cd $HOME/volta-bla
echo "Vilbert Results Train"

TASK_TYPE=active_passive
SPLIT_TYPE_RANDOM=finetune_random
# SPLIT_TYPE_TRAIN_GOLDEN=finetune_train_golden_large
# SPLIT_TYPE_TEST_GOLDEN=finetune_test_golden

ANNOTATIONS_PATH=/home/xchen/datasets/BLA_finetune/annotations
FEATURES_PATH=/home/xchen/datasets/BLA_finetune/imgfeats/volta-bla
TRAIN_BATCH_SIZE=16

VILBERT_MODEL=ctrl_vilbert
VILBERT_MODEL_CONFIG=ctrl_vilbert_base
VILBERT_PRETRAINED=/home/xchen/volta-bla/checkpoints/mmdata/${VILBERT_MODEL}/RetrievalMMdata_${VILBERT_MODEL_CONFIG}/pytorch_model_9.bin
VILBERT_OUTPUT_DIR=/home/xchen/volta-bla/exmaple_xinyi_bla_train/results/${VILBERT_MODEL}/active
FREEZE_BEFORE_LAYER=35

cd $HOME/volta-bla
echo "Vilbert Results Train"

python train_concap_modify.py \
    --annotations_path ${ANNOTATIONS_PATH}/${SPLIT_TYPE_RANDOM}/${TASK_TYPE} \
    --features_path ${FEATURES_PATH}/${SPLIT_TYPE_RANDOM}/${TASK_TYPE} \
    --config_file config/${VILBERT_MODEL_CONFIG}.json --from_pretrained ${VILBERT_PRETRAINED}\
    --train_batch_size ${TRAIN_BATCH_SIZE} --valid_batch_size 1 --num_train_epochs 10 \
    --freeze_before_layer ${FREEZE_BEFORE_LAYER} 

conda deactivate
