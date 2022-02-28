#!/bin/bash

ROOT='../../data/피부염' # data path
MODEL_PATH='../output/train/model/피부염/220224' # path where a model will be saved
LOG_PATH='../output/train/log/피부염/220224'
LOG_NAME='train2.log'
LOG_FREQ=100
WORKERS=4
EPOCHS=50
BATCH_SIZE=10
WEIGHT='[0.1,1.45,1.45]' # loss weight
LR='2e-04' # learning rate
WORLD_SIZE=1
RANK=0
PRETRAINED=' --pretrained' # or ''
DISTRIBUTED=' --gpu=1' # or ' --multiprocessing-distributed' #
NUM_EPOCHS_DECAY=30
LR_DECAY=0.1
SEED=42
# train
eval 'python train.py ${ROOT} --model-path=${MODEL_PATH} --log-path=${LOG_PATH} --log-name=${LOG_NAME} --log-freq=${LOG_FREQ} --workers=${WORKERS} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} --world-size=${WORLD_SIZE} --rank=${RANK} --loss-weight=${WEIGHT} -lr=${LR} --lr-decay=${LR_DECAY} --num-epochs-decay=${NUM_EPOCHS_DECAY} --seed=${SEED}${PRETRAINED}${DISTRIBUTED}'

LR='2e-05' # learning rate
# train
eval 'python train.py ${ROOT} --model-path=${MODEL_PATH} --log-path=${LOG_PATH} --log-name=${LOG_NAME} --log-freq=${LOG_FREQ} --workers=${WORKERS} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} --world-size=${WORLD_SIZE} --rank=${RANK} --loss-weight=${WEIGHT} -lr=${LR} --lr-decay=${LR_DECAY} --num-epochs-decay=${NUM_EPOCHS_DECAY} --seed=${SEED}${PRETRAINED}${DISTRIBUTED}'