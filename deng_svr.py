#!/usr/bin/env python3
#coding=utf-8

import time
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import pickle
from sklearn.svm import SVR

train_feature_path = sys.argv[1]
train_label_path = sys.argv[2]
test_feature_path = sys.argv[3]
submission_path = sys.argv[4]
prediction_path = sys.argv[5]

################
###   Util   ###
################
def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']
    df = df[features]
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add square terms
    new_features = ['reanalysis_specific_humidity_g_per_kg_2', 
                 'reanalysis_dew_point_temp_k_2', 
                 'station_avg_temp_c_2', 
                 'station_min_temp_c_2']
    for f in range(4):
        df[new_features[f]] = df[features[f]] ** 2

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq

def train_model(train, val):
    kernel = 'rbf'
    best_epsilon = 0.2

    x_train = train.iloc[:,0:8].values
    y_train = train.iloc[:,8].values
    x_val = val.iloc[:,0:8].values
    y_val = val.iloc[:,8].values

    grid = 0.1 * np.arange(1, 9, dtype=np.float64)
    best_epsilon = []
    best_score = 1000

    for epsilon in grid:
        model = SVR(kernel=kernel, epsilon=0.3)
        model.fit(x_train, y_train)

        predictions = model.predict(x_val).astype(int)
        score = eval_measures.meanabs(predictions, y_val)

        if score < best_score:
            best_epsilon = epsilon
            best_score = score

    print('validation epsilon: ', best_epsilon)
    print('validation score: ', best_score)

    full_data = pd.concat([train, val])
    x_data = full_data.iloc[:,0:8].values
    y_data = full_data.iloc[:,8].values
    model = SVR(kernel=kernel, epsilon=best_epsilon)
    model.fit(x_data, y_data)
    return model

def testing(data_path, sub_path, pred_path, sj_model, iq_model):
    sj_test, iq_test = preprocess_data(data_path)

    sj_predictions = sj_model.predict(sj_test.values).astype(int)
    iq_predictions = iq_model.predict(iq_test.values).astype(int)

    submission = pd.read_csv(sub_path, index_col=[0, 1, 2])

    submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
    submission.to_csv(pred_path)

    print('Testing is done.')

#########################
###   Main function   ###
#########################
def main():

    ### Read training data
    sj_train, iq_train = preprocess_data(train_feature_path, labels_path=train_label_path)

    ### Split validation
    sj_train = sj_train.sample(frac=1, replace=True)
    sj_train_subtrain = sj_train.head(842) 
    sj_train_subtest = sj_train.tail(sj_train.shape[0] - 842)

    iq_train = iq_train.sample(frac=1, replace=True)
    iq_train_subtrain = iq_train.head(468)
    iq_train_subtest = iq_train.tail(iq_train.shape[0] - 468)

    ### Training
    sj_svr_model = train_model(sj_train_subtrain, sj_train_subtest)
    iq_svr_model = train_model(iq_train_subtrain, iq_train_subtest)
    with open('sj_svr_model.pickle', 'wb') as sjm:
        pickle.dump(sj_svr_model, sjm)
    with open('iq_svr_model.pickle', 'wb') as iqm:
        pickle.dump(iq_svr_model, iqm)

    with open('sj_svr_16-202.pickle', 'rb') as sjm:
        sj_svr_model = pickle.load(sjm)
    with open('iq_svr_03-162.pickle', 'rb') as iqm:
        iq_svr_model = pickle.load(iqm)

    ### Testing
    testing(test_feature_path, submission_path, prediction_path, 
        sj_svr_model, iq_svr_model)

if __name__=='__main__':
    start_time = time.time()
    main()
    print('Elapse time:', time.time()-start_time, 'seconds\n')

