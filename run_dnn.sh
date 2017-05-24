#!/bin/bash
TRAIN_F=./train_feature.csv
TRAIN_L=./train_label.csv
TEST_F=./test_feature.csv
SUB=./submission_format.csv
PRED=./prediction_dnn.csv
# 0: sj, 1: all, 2: iq
MODE=1

python3.6 ./deng_dnn.py $TRAIN_F $TRAIN_L $TEST_F $SUB $PRED $MODE
