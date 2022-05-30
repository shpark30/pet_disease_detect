#!/bin/bash

ROOT='../../data/피부염' # data path
MODEL_PATH='../output/train/model/피부염/220224' 
LOG_PATH='../output/train/log/피부염/220224'
LOG_NAME='test.log'
TEST_RATIO=0.2
WORKERS=4
BATCH_SIZE=1
PRETRAINED=' --pretrained' # or ''
GPU='0' 
SEED=42

# test
for M in $MODEL_PATH/*
do
    eval 'python test.py ${ROOT} --model-path=${M} --log-path=${LOG_PATH} --log-name=${LOG_NAME} --test-ratio=${TEST_RATIO} --workers=${WORKERS} --batch_size=${BATCH_SIZE} --seed=${SEED} --gpu=${GPU}'
done

