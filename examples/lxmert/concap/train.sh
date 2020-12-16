#!/bin/bash

DATA=datasets/conceptual_captions
ANNOS=$DATA/annotations
FEATS=$DATA/resnet101_faster_rcnn_genome_imgfeats/volta
MODEL=lxmert
MODEL_CONFIG=lxmert
OUTPUT_DIR=checkpoints/conceptual_captions/${MODEL}
LOGGING_DIR=logs/conceptual_captions

source activate volta

cd ../../..
python train_concap.py \
  --bert_model bert-base-uncased --config_file config/${MODEL_CONFIG}.json \
  --train_batch_size 256 --gradient_accumulation_steps 1 --max_seq_length 20 \
  --learning_rate 1e-4 --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.05 --clip_grad_norm 1.0 \
  --objective 1 \
  --annotations_path $ANNOS --features_path $FEATS \
  --output_dir ${OUTPUT_DIR} \
  --logdir ${LOGGING_DIR} \
  --num_train_epochs 20 \
#  --resume_file ${OUTPUT_DIR}/${MODEL_CONFIG}/pytorch_ckpt_latest.tar

conda deactivate