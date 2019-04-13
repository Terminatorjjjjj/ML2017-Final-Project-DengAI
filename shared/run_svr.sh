#!/bin/bash
TRAIN_F=./train_feature.csv
TRAIN_L=./train_label.csv
TEST_F=./test_feature.csv
SUB=./submission_format.csv
PRED=./prediction_svr.csv

python3.6 ./deng_svr.py $TRAIN_F $TRAIN_L $TEST_F $SUB $PRED