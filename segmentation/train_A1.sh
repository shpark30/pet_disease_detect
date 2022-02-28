#!/bin/bash

ROOT='../../data/A2_비듬_각질_상피성잔고리/' # data path
MODEL_PATH='../output/train/model/A2_비듬_각질_상피성잔고리/220225' # path where a model will be saved
LOG_PATH='../output/train/log/A2_비듬_각질_상피성잔고리/220225'
LOG_NAME='train1.log'
LOG_FREQ=1
WORKERS=4
EPOCHS=10
BATCH_SIZE=8
WEIGHT='[1.,1.,1.]' # loss weight
LR='2e-05' # learning rate
WORLD_SIZE=1
RANK=0
PRETRAINED=' --pretrained' # or ''
DISTRIBUTED=' --gpu=0' # or ' --multiprocessing-distributed' #
NUM_EPOCHS_DECAY=30
LR_DECAY=0.1
SEED=42

# train
eval 'python train.py ${ROOT} --model-path=${MODEL_PATH} --log-path=${LOG_PATH} --log-name=${LOG_NAME} --log-freq=${LOG_FREQ} --workers=${WORKERS} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} --world-size=${WORLD_SIZE} --rank=${RANK} --loss-weight=${WEIGHT} -lr=${LR} --lr-decay=${LR_DECAY} --num-epochs-decay=${NUM_EPOCHS_DECAY} --seed=${SEED}${PRETRAINED}${DISTRIBUTED}'