#!/bin/bash

ROOT='../../../input/data/구진플라크' # data path
MODEL_PATH='../../../output/train/model/구진플라크/220221' # path where a model will be saved
LOG_PATH='../../../output/train/log/구진플라크/220221'
LOG_NAME='train.log'
LOG_FREQ=1
WORKERS=1
EPOCHS=1
BATCH_SIZE=6
WEIGHT='[1.,1.,1.]' # loss weight
LR='2e-05' # learning rate
WORLD_SIZE=1
RANK=0
PRETRAINED=' --pretrained' # or ''
DISTRIBUTED=' --gpu=0' # or ' --multiprocessing-distributed' #
NUM_EPOCHS_DECAY=30
LR_DECAY=0.1

# train
eval 'python train.py ${ROOT} --model-path=${MODEL_PATH} --log-path=${LOG_PATH} --log-name=${LOG_NAME} --log-freq=${LOG_FREQ} --workers=${WORKERS} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} --world-size=${WORLD_SIZE} --rank=${RANK} --loss-weight=${WEIGHT} -lr=${LR} --lr-decay=${LR_DECAY} --num-epochs-decay=${NUM_EPOCHS_DECAY}${PRETRAINED}${DISTRIBUTED}'
