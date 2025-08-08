#!/bin/bash

MODEL=deeplabv3bga_resnet101
DATA_ROOT=datasets/VOCdevkit/VOC2012
DATASET=voc
TASK=10-1
EPOCH=5
BATCH=16
LOSS=bce_loss
LR=0.001
THRESH=0.7
MEMORY=0

SUBPATH=KProBA
CURR=0

now=$(date +"%Y%m%d_%H%M%S")
result_dir=./checkpoints/${SUBPATH}/${TASK}/

if [ ! -d ${result_dir} ]; then
    mkdir -p ${result_dir}
fi

# Set number of GPUs
NUM_GPUS=1

# Use torchrun for multi-GPU training
torchrun --nproc_per_node=${NUM_GPUS} train.py \
    --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --bn_freeze --amp \
    --curr_step ${CURR} --subpath ${SUBPATH} --initial \
    --overlap \
    | tee ${result_dir}/train-$now.log
