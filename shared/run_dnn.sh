#!/bin/bash
TRAIN_F=./data/train_feature.csv
TRAIN_L=./data/train_label.csv
TEST_F=./data/test_feature.csv
SUB=./data/submission_format.csv
PRED=./prediction_dnn.csv

python3.6 ./deng_dnn.py $TRAIN_F $TRAIN_L $TEST_F $SUB $PRED
