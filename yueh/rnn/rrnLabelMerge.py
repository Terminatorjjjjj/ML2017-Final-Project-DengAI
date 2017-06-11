#!/usr/bin/env python3
#coding=utf-8

import numpy as np
import pandas as pd
import sys
import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import regularizers
import h5py
import matplotlib.pyplot as plt
import pickle
import keras.backend as K
from keras.models import load_model
import pickle

start_time = time.time()
"""
train_feature_file = sys.argv[1]
train_label_file = sys.argv[2]
test_feature_file = sys.argv[3]
result_path = sys.argv[4]
model_sj_path = sys.argv[5]
mean_sj_path = sys.argv[6]
std_sj_path = sys.argv[7]
model_iq_path = sys.argv[8]
mean_iq_path = sys.argv[9]
std_iq_path = sys.argv[10]
"""
train_feature_file = 'train_feature.csv'
train_label_file = 'train_label.csv'
test_feature_file = 'test_feature.csv'
arg = False
"""
if arg == True:
    result_path = sys.argv[1]
    weekConcat = int(sys.argv[2])
    model_sj_path = str(sys.argv[3])
    model_iq_path = str(sys.argv[4])
    add = int(sys.argv[5])
else:
    result_path = 'pred.csv'
    weekConcat = 52 
    model_sj_path = 'labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_52_3_.h5py'
    model_iq_path = 'labeliq32_3000_100_adam_elu_softmax_1iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_52_3_.h5py'
    add = 3
"""
def reshape(train_feature,weekConcat):
    trainRow = train_feature.shape[0]-(weekConcat)
    col = train_feature.shape[1]
    trainF = np.zeros((trainRow, weekConcat, col)) 
    for i in range(0,trainRow):
        trainF[i] =  train_feature[i:i+weekConcat,:]  
    return trainF

def main(model_sj_path, model_iq_path, add,weekConcat):
    #-----Read training data---------------------------------------------------------
    train_feature = pd.read_csv(train_feature_file, encoding='big5')
    train_feature = train_feature.values
    train_feature = train_feature.astype(str)
    train_feature[train_feature == 'nan'] = '0.0'
    train_feature = np.hstack((train_feature[:,2].reshape(len(train_feature),1), train_feature[:,4::]))
    train_feature = train_feature.astype(float)
    #-----Read training label--------------------------------------------------------
    train_label = pd.read_csv(train_label_file, encoding='big5')
    train_label = train_label.values
    train_label = train_label[:,3].reshape(len(train_label),1).astype(float)
    
    #-----Read testing data---------------------------------------------------------
    test_feature = pd.read_csv(test_feature_file, encoding='big5')
    test_feature = test_feature.values
    test_feature = test_feature.astype(str)
    test_feature[test_feature == 'nan'] = '0.0'
    test_feature = np.hstack((test_feature[:,2].reshape(len(test_feature),1), test_feature[:,4::]))
    test_feature = test_feature.astype(float)
    
    if add == 1:
        train_feature = np.hstack((train_feature, np.power(train_feature[:,[8,10,14,16,19]],3)))
        test_feature = np.hstack((test_feature,np.power(test_feature[:,[8,10,14,16,19]],3)))
    elif add == 2:
        train_feature = np.hstack((train_feature, np.square(train_feature)))
        test_feature = np.hstack((test_feature, np.square(test_feature)))
    elif add == 3:
        train_feature = train_feature[:,[8,14,16,19]]
        test_feature = test_feature[:,[8,14,16,19]]
    elif add == 4:
        train_feature = np.hstack((train_feature[:,[8,14,16,19]], np.square(train_feature[:,[8,14,16,19]])))
        test_feature = np.hstack((test_feature[:,[8,14,16,19]], np.square(test_feature[:,[8,14,16,19]])))
    
    ## Seperate diff city
    train_sj_f = train_feature[:936]
    train_sj_l = train_label[:936]
    train_iq_f = train_feature[936::]
    train_iq_l = train_label[936::]
    
    test_sj = test_feature[:260]
    test_iq = test_feature[260::]
    test_sj_fakeLabel = np.zeros((test_sj.shape[0],1))
    test_iq_fakeLabel = np.zeros((test_iq.shape[0],1))
    
    
    ## integrate train & test + count mean & std    
    all_feature_sj = np.vstack((train_sj_f, test_sj))    
    all_feature_iq = np.vstack((train_iq_f, test_iq))
    mean_sj = np.mean(all_feature_sj, axis=0)
    std_sj = np.std(all_feature_sj, axis=0)
    mean_iq = np.mean(all_feature_iq, axis=0)
    std_iq = np.std(all_feature_iq, axis=0)
    mean_sj = np.append(mean_sj,(np.mean(train_sj_l)))    
    std_sj = np.append(std_sj,(np.std(train_sj_l)))
    mean_iq = np.append(mean_iq,(np.mean(train_iq_l)))    
    std_iq = np.append(std_iq,(np.std(train_iq_l)))    
    
    ## integrate label & fakeLabel
    test_sj = np.vstack((train_sj_f[train_sj_f.shape[0]-weekConcat:],test_sj))
    test_iq = np.vstack((train_iq_f[train_iq_f.shape[0]-weekConcat:],test_iq))
    test_sj_fakeLabel = np.vstack((train_sj_l[train_sj_l.shape[0]-weekConcat:],test_sj_fakeLabel))
    test_iq_fakeLabel = np.vstack((train_iq_l[train_iq_l.shape[0]-weekConcat:],test_iq_fakeLabel))
    test_sj = np.hstack((test_sj,test_sj_fakeLabel))
    test_iq = np.hstack((test_iq,test_iq_fakeLabel))
    
    ## reshape
    test_sj = reshape(test_sj,weekConcat)
    test_iq = reshape(test_iq,weekConcat)
    
    ## normalization
    test_sj = (test_sj - mean_sj) / std_sj
    test_iq = (test_iq - mean_iq) / std_iq
    
      
    ## pred
    y_pred_sj = np.zeros((test_sj.shape[0]))
    y_pred_iq = np.zeros((test_iq.shape[0]))
    model_sj = load_model(model_sj_path)
    model_iq = load_model(model_iq_path)
    rowsj = test_sj.shape[0]
    rowiq = test_iq.shape[0]
    for i in range(0,rowsj):
        y_pred_sj[i] = model_sj.predict(np.reshape(test_sj[i],(1,test_sj.shape[1],test_sj.shape[2])))[0][0]
        normPred = (y_pred_sj[i] - mean_sj[mean_sj.shape[0]-1])/std_sj[std_sj.shape[0]-1]
        for j in range(1, weekConcat+1):
            if (j+i) < test_sj.shape[0]:
                test_sj[j+i,(weekConcat-j),(std_sj.shape[0]-1)] = normPred
                
    for i in range(0,rowiq):
        y_pred_iq[i] = model_iq.predict(np.reshape(test_iq[i],(1,test_iq.shape[1],test_iq.shape[2])))[0][0]
        normPred = (y_pred_iq[i] - mean_iq[mean_iq.shape[0]-1])/std_iq[std_iq.shape[0]-1]
        for j in range(1, weekConcat+1):
            if (j+i) < test_iq.shape[0]:
                test_iq[j+i,(weekConcat-j),(std_iq.shape[0]-1)] = normPred
        
        
    
    y_pred = np.vstack((np.reshape(y_pred_sj,(y_pred_sj.shape[0],1)), np.reshape(y_pred_iq,(y_pred_iq.shape[0],1))))
    return y_pred


if __name__ == "__main__":
    test_feature = pd.read_csv(test_feature_file, encoding='big5')
    test_feature = test_feature.values
    test_feature = test_feature.astype(str)
    test_tags = np.asarray(test_feature[:,:3])
    
    weekConcat1 = 10
    model_sj_path1 = 'labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_3_.h5py'
    model_iq_path1 = 'labeliq32_3000_100_adam_elu_softmax_1iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_10_3_.h5py'
    add1 = 3
    weekConcat2 = 10 
    model_sj_path2 = 'labelsj32_3000_100_adam_elu_softmax_2sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_4_.h5py'
    model_iq_path2 = 'labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.52_256_0.4_1280.4_64_0.4_10_4_.h5py'
    add2 = 4
    weekConcat3 = 10
    model_sj_path3 = 'labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_10_3_.h5py'
    model_iq_path3 = 'labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_3_.h5py'
    add3 = 3
    weekConcat4 = 52 
    model_sj_path4 = 'labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_52_3_.h5py'
    model_iq_path4 = 'labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_52_3_.h5py'
    add4 = 3
    weekConcat5 = 12
    model_sj_path5 = 'labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_12_4_.h5py'
    model_iq_path5 = 'labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.52_256_0.4_1280.4_64_0.4_12_4_.h5py'
    add5 = 4 
    y_pred1 = main(model_sj_path1, model_iq_path1, add1,weekConcat1)
    y_pred2 = main(model_sj_path2, model_iq_path2, add2,weekConcat2)
    y_pred3 = main(model_sj_path3, model_iq_path3, add3,weekConcat3)
    y_pred4 = main(model_sj_path4, model_iq_path4, add4,weekConcat4)
    y_pred5 = main(model_sj_path5, model_iq_path5, add5,weekConcat5)
    y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5)/5
    
    output = open("pred_mix.csv", 'w')
    output.write('city,year,weekofyear,total_cases\n')
    for i in range(len(y_pred)):
        line = ''
        for j in range(3):
            line += str(test_tags[i,j]) + ','
        tmp = abs(round(float(y_pred[i])))
        line += str(int(tmp)) + '\n'
        output.write(line)
    output.close()
    
    print('Elapse time:', time.time()-start_time, 'seconds\n')
    