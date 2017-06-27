#!/usr/bin/env python3.6
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
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

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
    weekPredAfter = int(sys.argv[6])
else:
    result_path = 'pred.csv'
    weekConcat = 52 
    model_sj_path = 'labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_52_3_.h5py'
    model_iq_path = 'labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_52_3_.h5py'
    add = 3
    weekPredAfter = 5
"""
def reshape(train_feature,weekConcat,weekPredAfter):
    trainRow = train_feature.shape[0]-(weekConcat-1) -(weekPredAfter)
    col = train_feature.shape[1]
    trainF = np.zeros((trainRow, weekConcat, col)) 
    for i in range(0,trainRow):
        trainF[i] =  train_feature[i:i+weekConcat,:]  
    return trainF


def main(model_sj_path, model_iq_path, add,weekConcat,weekPredAfter):
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
    test_sj = np.vstack((train_sj_f[train_sj_f.shape[0]-((weekConcat-1)+weekPredAfter):],test_sj))
    test_iq = np.vstack((train_iq_f[train_iq_f.shape[0]-((weekConcat-1)+weekPredAfter):],test_iq))
    test_sj_fakeLabel = np.vstack((train_sj_l[train_sj_l.shape[0]-((weekConcat-1)+weekPredAfter):],test_sj_fakeLabel))
    test_iq_fakeLabel = np.vstack((train_iq_l[train_iq_l.shape[0]-((weekConcat-1)+weekPredAfter):],test_iq_fakeLabel))
    test_sj = np.hstack((test_sj,test_sj_fakeLabel))
    test_iq = np.hstack((test_iq,test_iq_fakeLabel))
    
    ## reshape
    test_sj = reshape(test_sj,weekConcat,weekPredAfter)
    test_iq = reshape(test_iq,weekConcat,weekPredAfter)
    
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
        for j in range(weekPredAfter, weekConcat+weekPredAfter):
            if (j+i) < test_sj.shape[0]:
                test_sj[j+i,(weekConcat-j),(std_sj.shape[0]-1)] = normPred
                
    for i in range(0,rowiq):
        y_pred_iq[i] = model_iq.predict(np.reshape(test_iq[i],(1,test_iq.shape[1],test_iq.shape[2])))[0][0]
        normPred = (y_pred_iq[i] - mean_iq[mean_iq.shape[0]-1])/std_iq[std_iq.shape[0]-1]
        for j in range(weekPredAfter, weekConcat+weekPredAfter):
            if (j+i) < test_iq.shape[0]:
                test_iq[j+i,(weekConcat-j),(std_iq.shape[0]-1)] = normPred
        
    y_pred = np.vstack((np.reshape(y_pred_sj,(y_pred_sj.shape[0],1)), np.reshape(y_pred_iq,(y_pred_iq.shape[0],1))))
    return y_pred


if __name__=='__main__':
    test_feature = pd.read_csv(test_feature_file, encoding='big5')
    test_feature = test_feature.values
    test_feature = test_feature.astype(str)
    test_tags = np.asarray(test_feature[:,:3])
    """
    weekConcat1 = 10
    model_sj_path1 = 'saveModel/2295/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_3_.h5py'
    model_iq_path1 = 'saveModel/2295/labeliq32_3000_100_adam_elu_softmax_1iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_10_3_.h5py'
    add1 = 3
    weekPredAfter1 = 1
    weekConcat2 = 10 
    model_sj_path2 = 'saveModel/2295/labelsj32_3000_100_adam_elu_softmax_2sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_4_.h5py'
    model_iq_path2 = 'saveModel/2295/labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.52_256_0.4_1280.4_64_0.4_10_4_.h5py'
    add2 = 4
    weekPredAfter2 = 1
    weekConcat3 = 10
    model_sj_path3 = 'saveModel/2295/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_10_3_.h5py'
    model_iq_path3 = 'saveModel/2295/labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_3_.h5py'
    add3 = 3
    weekPredAfter3 = 1
    """
    """
    weekConcat4 = 52 
    model_sj_path4 = 'saveModel/2295/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_52_3_.h5py'
    model_iq_path4 = 'saveModel/2295/labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_52_3_.h5py'
    add4 = 3
    weekPredAfter4 = 5
    """
    """
    weekConcat5 = 12
    model_sj_path5 = 'saveModel/2295/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_12_4_.h5py'
    model_iq_path5 = 'saveModel/2295/labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.52_256_0.4_1280.4_64_0.4_12_4_.h5py'
    add5 = 4 
    weekPredAfter5 = 1
    
    y_pred1 = main(model_sj_path1, model_iq_path1, add1,weekConcat1,weekPredAfter1)
    y_pred2 = main(model_sj_path2, model_iq_path2, add2,weekConcat2,weekPredAfter2)
    y_pred3 = main(model_sj_path3, model_iq_path3, add3,weekConcat3,weekPredAfter3)
    #y_pred4 = main(model_sj_path4, model_iq_path4, add4,weekConcat4,weekPredAfter4)
    y_pred5 = main(model_sj_path5, model_iq_path5, add5,weekConcat5,weekPredAfter5)
    y_pred = (y_pred1 + y_pred2 + y_pred3 +  y_pred5)/4
    """
    y_pred1 = main('sj_10_3_1_1-62-10.42.hdf5','iq_10_3_1_1-38-0.39.hdf5',3,10,1)
    y_pred2 = main('sj_10_3_1_2-26-10.09.hdf5','iq_10_3_1_2-40-0.76.hdf5',3,10,1)
    y_pred3 = main('sj_10_3_2_1-14-10.65.hdf5','iq_10_3_2_1-54-0.74.hdf5',3,10,1)
    y_pred4 = main('sj_10_3_2_2-18-9.93.hdf5','iq_10_3_2_2-28-0.40.hdf5',3,10,1)
    y_pred5 = main('sj_10_3_3_1-23-10.61.hdf5','iq_10_3_3_1-20-0.47.hdf5',3,10,1)
    y_pred6 = main('sj_10_3_3_2-20-11.13.hdf5','iq_10_3_3_2-114-0.63.hdf5',3,10,1)
    y_pred7 = main('sj_10_4_1_1-14-10.30.hdf5','iq_10_4_1_1-27-0.65.hdf5',4,10,1)
    y_pred8 = main('sj_10_4_1_2-16-9.50.hdf5','iq_10_4_1_2-154-0.85.hdf5',4,10,1)
    y_pred9 = main('sj_10_4_2_1-33-10.38.hdf5','iq_10_4_2_1-47-0.75.hdf5',4,10,1)
    y_pred10 = main('sj_10_4_2_2-17-9.95.hdf5','iq_10_4_2_2-38-0.87.hdf5',4,10,1)
    y_pred11 = main('sj_10_4_3_1-28-10.66.hdf5','iq_10_4_3_1-145-0.98.hdf5',4,10,1)
    y_pred12 = main('sj_10_4_3_2-21-10.86.hdf5','iq_10_4_3_2-63-0.62.hdf5',4,10,1)

    y_pred13 = main('sj_12_3_1_1-13-9.83.hdf5','iq_12_3_1_1-33-0.74.hdf5',3,12,1)
    y_pred14 = main('sj_12_3_1_2-12-9.37.hdf5','iq_12_3_1_2-39-0.58.hdf5',3,12,1)
    y_pred15 = main('sj_12_3_2_1-22-10.29.hdf5','iq_12_3_2_1-12-0.89.hdf5',3,12,1)
    y_pred16 = main('sj_12_3_2_2-14-9.91.hdf5','iq_12_3_2_2-13-0.82.hdf5',3,12,1)
    y_pred17 = main('sj_12_3_3_1-13-10.28.hdf5','iq_12_3_3_1-63-0.79.hdf5',3,12,1)
    y_pred18 = main('sj_12_3_3_2-25-10.63.hdf5','iq_12_3_3_2-60-0.63.hdf5',3,12,1)
    y_pred19 = main('sj_12_4_1_1-56-10.07.hdf5','iq_12_4_1_1-04-0.81.hdf5',4,12,1)
    y_pred20 = main('sj_12_4_1_2-30-9.88.hdf5','iq_12_4_1_2-05-0.75.hdf5',4,12,1)
    y_pred21 = main('sj_12_4_2_1-41-10.55.hdf5','iq_12_4_2_2-43-0.73.hdf5',4,12,1)
    y_pred22 = main('sj_12_4_2_2-73-10.64.hdf5','iq_12_4_2_1-20-0.91.hdf5',4,12,1)
    y_pred23 = main('sj_12_4_3_1-16-11.42.hdf5','iq_12_4_3_1-30-0.77.hdf5',4,12,1)
    y_pred24 = main('sj_12_4_3_2-17-10.29.hdf5','iq_12_4_3_2-85-0.63.hdf5',4,12,1)
    
    y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5 + y_pred6 + y_pred7 + y_pred8 +
              y_pred9 + y_pred10 + y_pred11 + y_pred12 + y_pred13 + y_pred14 + y_pred15 + y_pred16+
              y_pred17 + y_pred18 + y_pred19 + y_pred20 + y_pred21 + y_pred22 + y_pred23 + y_pred24)/24
    output = open("pred_mix24.csv", 'w')
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
    
