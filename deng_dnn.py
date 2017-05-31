#!/usr/bin/env python3
#coding=utf-8

import time
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import regularizers
from keras.models import load_model
import h5py

train_feature_path = sys.argv[1]
train_label_path = sys.argv[2]
test_feature_path = sys.argv[3]
submission_path = sys.argv[4]
prediction_path = sys.argv[5]
# mode = sys.argv[6]
mode = 0

################
###   Util   ###
################
def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    core_features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']
    features = ['precipitation_amt_mm',
                'reanalysis_air_temp_k',
                'reanalysis_avg_temp_k',
                'reanalysis_dew_point_temp_k',
                'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k',
                'reanalysis_precip_amt_kg_per_m2',
                'reanalysis_relative_humidity_percent',
                'reanalysis_sat_precip_amt_mm',
                'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_tdtr_k',
                'station_avg_temp_c',
                'station_diur_temp_rng_c',
                'station_max_temp_c',
                'station_min_temp_c',
                'station_precip_mm']
    df = df[features]
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # # add square terms
    new_features = ['reanalysis_specific_humidity_g_per_kg_2', 
                 'reanalysis_dew_point_temp_k_2', 
                 'station_avg_temp_c_2', 
                 'station_min_temp_c_2']
    for f in range(4):
        df[new_features[f]] = df[core_features[f]] ** 2

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    # print(df.iloc[:,-1].max())
    # print(df.iloc[:,-1].min())
    # normalize
    tmp = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        tmp[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    df = tmp
    del tmp
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    return sj, iq

def train_sj_model(save_path, subtrain, subtest):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(subtrain.shape[1]-1,)))
    # model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    model.summary()

    model.compile(loss='mean_absolute_error', optimizer='adam')

    es = EarlyStopping(monitor='val_loss', patience=150, verbose=1, mode='min')
    ck = ModelCheckpoint(filepath=save_path, 
                         verbose=1,
                         save_best_only=True,
                         save_weights_only=False,
                         monitor='val_loss',
                         mode='min')

    x_train = subtrain.iloc[:,0:-1].values
    y_train = subtrain.iloc[:,-1].values
    x_val = subtest.iloc[:,0:-1].values
    y_val = subtest.iloc[:,-1].values
    h = model.fit(x_train, y_train, epochs=3000, batch_size=64, #16
                     validation_data=(x_val, y_val), 
                     callbacks=[es, ck])

    print('sj model min loss: ', min(h.history['val_loss']))
    return model

def train_iq_model(save_path, subtrain, subtest):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(subtrain.shape[1]-1,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    model.summary()

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')
    ck = ModelCheckpoint(filepath=save_path, 
                         verbose=1,
                         save_best_only=True,
                         save_weights_only=False,
                         monitor='val_loss',
                         mode='min')

    x_train = subtrain.iloc[:,0:-1].values
    y_train = subtrain.iloc[:,-1].values
    x_val = subtest.iloc[:,0:-1].values
    y_val = subtest.iloc[:,-1].values
    h = model.fit(x_train, y_train, epochs=3000, batch_size=64,
                     validation_data=(x_val, y_val), 
                     callbacks=[es, ck])

    print('iq model min loss: ', min(h.history['val_loss']))
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
    if mode <= 1:
        sj_best_model = train_sj_model('sj_model.h5py', sj_train_subtrain, sj_train_subtest)
    if mode >= 1:
        iq_best_model = train_iq_model('iq_model.h5py', iq_train_subtrain, iq_train_subtest)

    # sj_best_model = load_model('sj_17-261.h5py')
    # iq_best_model = load_model('iq_03-558.h5py')
    ### Testing
    if mode == 1:
        testing(test_feature_path, submission_path, prediction_path, 
            sj_best_model, iq_best_model)

if __name__=='__main__':
    start_time = time.time()
    main()
    print('Elapse time:', time.time()-start_time, 'seconds\n')

