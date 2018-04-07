#!/usr/bin/env python2.7
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
    weekConcat = int(sys.argv[1])
    model_sj_path = str(sys.argv[2])
    model_iq_path = str(sys.argv[3])
    add = int(sys.argv[4])
    weekPredAfter = int(sys.argv[5])
else:
    weekConcat = 10 
    model_sj_path = 'saveModel/2295/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_3_.h5py'
    model_iq_path = 'saveModel/2295/labeliq32_3000_100_adam_elu_softmax_1iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_10_3_.h5py'
    add = 3
    weekPredAfter = 1 
"""
def reshape(train_feature,weekConcat,weekPredAfter):
    trainRow = train_feature.shape[0]-(weekConcat-1) -(weekPredAfter)
    col = train_feature.shape[1]
    trainF = np.zeros((trainRow, weekConcat, col)) 
    for i in range(0,trainRow):
        trainF[i] =  train_feature[i:i+weekConcat,:]  
    return trainF

def drawTrain(train_sj_f, train_sj_l, train_iq_f, train_iq_l,weekConcat,weekPredAfter, add, model_sj_path, model_iq_path, mean_sj, std_sj, mean_iq, std_iq):
    ## peak label
    peak = np.hstack((np.arange(15,38),np.arange(67,90)))
    peak = np.hstack((peak,np.arange(119,143)))
    peak = np.hstack((peak,np.arange(172,195)))
    peak = np.hstack((peak,np.arange(224,247)))
    peak = np.hstack((peak,np.arange(276,298)))
    peak = np.hstack((peak,np.arange(327,350)))
    peak = np.hstack((peak,np.arange(379,402)))
    peak = np.hstack((peak,np.arange(428,455)))
    peak = np.hstack((peak,np.arange(485,507)))
    peak = np.hstack((peak,np.arange(534,558)))
    peak = np.hstack((peak,np.arange(587,610)))
    peak = np.hstack((peak,np.arange(639,662)))
    peak = np.hstack((peak,np.arange(691,714)))
    peak = np.hstack((peak,np.arange(743,767)))
    peak = np.hstack((peak,np.arange(796,819)))
    peak = np.hstack((peak,np.arange(848,870)))
    peak = np.hstack((peak,np.arange(899,922)))
    peak = peak - 2 - weekConcat - weekPredAfter
    
    
    log = str(add) + '_' + str(weekConcat) + '_' + str(weekPredAfter)
    ## join data feature & label
    train_sj_f = np.hstack((train_sj_f, train_sj_l))
    train_iq_f = np.hstack((train_iq_f, train_iq_l))
    ## normalization
    train_sj_f = (train_sj_f - mean_sj) / std_sj
    train_iq_f = (train_iq_f - mean_iq) / std_iq
    ## reshape
    train_sj_f = reshape(train_sj_f,weekConcat,weekPredAfter)
    train_iq_f = reshape(train_iq_f,weekConcat,weekPredAfter)   
    ## predict & load
    model_sj = load_model(model_sj_path)
    model_iq = load_model(model_iq_path)
    predSj = model_sj.predict(train_sj_f)
    predIq = model_iq.predict(train_iq_f)
    ## Draw
    originLabel = np.vstack((train_sj_l[weekConcat:], train_iq_l[weekConcat:]))
    pred = np.vstack((predSj,predIq))
    
    fig = plt.figure(figsize=(28, 16), dpi=80)
    plt.plot(train_sj_l[weekConcat:])
    plt.plot(predSj)
    mae = np.sum(np.absolute((train_sj_l[weekConcat:] - predSj)))/predSj.size
    peakMae = np.sum(np.absolute(((train_sj_l[weekConcat:])[peak] - predSj[peak])))/peak.size
    plt.annotate('MAE = %s' % mae, xy=(2, 1), xytext=(30, 311.5))
    plt.annotate('PEAK_MAE = %s' % peakMae, xy=(2, 1), xytext=(30, 361.5))
    #plt.show()
    fig.savefig(log+'.png')
    
def drawPeak():
    ## peak label
    peak = np.hstack((np.arange(15,38),np.arange(67,90)))
    peak = np.hstack((peak,np.arange(120,143)))
    peak = np.hstack((peak,np.arange(172,195)))
    peak = np.hstack((peak,np.arange(224,245)))
    peak = peak - 2
    return peak


def main(model_sj_path, model_iq_path, add,weekConcat,weekPredAfter):
    #-----Read training data---------------------------------------------------------
    train_feature = pd.read_csv(train_feature_file, encoding='big5')
    if weekPredAfter == 2:
        train_feature.fillna(method='ffill', inplace=True)
        train_feature = train_feature.values
        train_feature = train_feature.astype(str)
    else:
        #train_feature.fillna(method='ffill', inplace=True)
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
    if weekPredAfter == 2:
        test_feature.fillna(method='ffill', inplace=True)
        test_feature = test_feature.values
        test_feature = test_feature.astype(str)
    else:
        test_feature = test_feature.values
        test_feature = test_feature.astype(str)
        test_feature[test_feature == 'nan'] = '0.0'
    test_feature = np.hstack((test_feature[:,2].reshape(len(test_feature),1), test_feature[:,4::]))
    test_feature = test_feature.astype(float)
    
    
    
    
    
    weekPredAfter = 1
    
    
    
    
    
    
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
    elif add == 5:
        train_feature[:,0] = np.abs(np.cos((train_feature[:,0]-1)/52 * np.pi))
        test_feature[:,0] = np.abs(np.cos((test_feature[:,0]-1)/52 * np.pi))
        train_feature = train_feature[:,[0,8,14,16,19]]
        test_feature = test_feature[:,[0,8,14,16,19]]
    elif add == 6:
        train_feature[:,0] = np.abs(np.cos((train_feature[:,0]-1)/52 * np.pi))
        test_feature[:,0] = np.abs(np.cos((test_feature[:,0]-1)/52 * np.pi))
        train_feature = np.hstack((train_feature[:,[0,8,14,16,19]], np.square(train_feature[:,[8,14,16,19]])))
        test_feature = np.hstack((test_feature[:,[0,8,14,16,19]], np.square(test_feature[:,[8,14,16,19]])))
        
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
    
    #drawTrain(train_sj_f, train_sj_l, train_iq_f, train_iq_l,weekConcat,weekPredAfter, add, model_sj_path, model_iq_path, mean_sj, std_sj, mean_iq, std_iq)
     
    ## pred
    y_pred_sj = np.zeros((test_sj.shape[0]))
    y_pred_iq = np.zeros((test_iq.shape[0]))
    model_sj = load_model(model_sj_path)
    model_iq = load_model(model_iq_path)
    rowsj = test_sj.shape[0]
    rowiq = test_iq.shape[0]
    for i in range(0,rowsj):
        y_pred_sj[i] = model_sj.predict(np.reshape(test_sj[i],(1,test_sj.shape[1],test_sj.shape[2])))[0][0]
        """
        if np.size(np.where(i == peak)[0]) == 1:
            y_pred_sj[i] = y_pred_sj[i] + 0.005 * y_pred_sj[i]**2 
        """
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
    


if __name__ == "__main__":
    test_feature = pd.read_csv(test_feature_file, encoding='big5')
    test_feature = test_feature.values
    test_feature = test_feature.astype(str)
    test_tags = np.asarray(test_feature[:,:3])
    
    weekConcat1 = 10
    model_sj_path1 = 'saveModel/2221/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_3_.h5py'
    model_iq_path1 = 'saveModel/2221/labeliq32_3000_100_adam_elu_softmax_1iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_10_3_.h5py'
    add1 = 3
    weekPredAfter1 = 1
    weekConcat2 = 10 
    model_sj_path2 = 'saveModel/2221/labelsj32_3000_100_adam_elu_softmax_2sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_4_.h5py'
    model_iq_path2 = 'saveModel/2221/labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.52_256_0.4_1280.4_64_0.4_10_4_.h5py'
    add2 = 4
    weekPredAfter2 = 1
    weekConcat3 = 10
    model_sj_path3 = 'saveModel/2221/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_10_3_.h5py'
    model_iq_path3 = 'saveModel/2221/labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_10_3_.h5py'
    add3 = 3
    weekPredAfter3 = 1
    
    """
    weekConcat4 = 52 
    model_sj_path4 = 'saveModel/2221/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.53_256_0.4_1280.4_64_0.4_52_3_.h5py'
    model_iq_path4 = 'saveModel/2221/labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_52_3_.h5py'
    add4 = 3
    weekPredAfter4 = 5
    """
    
    weekConcat5 = 12
    model_sj_path5 = 'saveModel/2221/labelsj32_3000_100_adam_elu_softmax_1sj_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.51_256_0.4_1280.4_64_0.4_12_4_.h5py'
    model_iq_path5 = 'saveModel/2221/labeliq32_3000_100_adam_elu_softmax_2iq_256_512_256_1280.5_0.7_0.5_0.332_32_64_320.1_0.5_0.5_0.52_256_0.4_1280.4_64_0.4_12_4_.h5py'
    add5 = 4 
    weekPredAfter5 = 1
    
    y_pred1 = main(model_sj_path1, model_iq_path1, add1,weekConcat1,weekPredAfter1)
    y_pred2 = main(model_sj_path2, model_iq_path2, add2,weekConcat2,weekPredAfter2)
    y_pred3 = main(model_sj_path3, model_iq_path3, add3,weekConcat3,weekPredAfter3)
    #y_pred4 = main(model_sj_path4, model_iq_path4, add4,weekConcat4,weekPredAfter4)
    y_pred5 = main(model_sj_path5, model_iq_path5, add5,weekConcat5,weekPredAfter5)
    y_pred4 = main('saveModel/2221/sj_5_5_3_2_1-210-17.80.hdf5','saveModel/2221/iq_5_5_3_2_1-126-0.68.hdf5',5,5,2)
    #y_pred4 = (pd.read_csv('best_1.csv', encoding='big5')).values[:,3]
    peak = drawPeak()
    y_pred = (y_pred1 + y_pred2 + y_pred3 +  y_pred5)/4
    
    idx = 0
    i = 0
    while i <= peak[peak.shape[0]-1]:
        if i == peak[idx]:
            y_pred[i] = (y_pred[i] * 4 + y_pred4[i] * 1)/5
            #y_pred[i] = y_pred4[i]
            idx = idx + 1
        i = i + 1
    
    
    """
    y_pred1 = main('saveModel/sj_5_5_1_2_1-35-9.32.hdf5','saveModel/iq_5_5_1_2_1-33-0.66.hdf5',5,5,2)
    y_pred2 = main('saveModel/sj_5_6_1_2_1-23-9.62.hdf5','saveModel/iq_5_6_1_2_1-00-0.82.hdf5',6,5,2)
    #y_pred3 = main('saveModel/sj_5_5_3_2_1-28-10.49.hdf5','saveModel/iq_5_5_3_2_1-126-0.68.hdf5',5,5,2)
    #y_pred4 = main('saveModel/sj_5_6_3_2_1-26-10.12.hdf5','saveModel/iq_5_6_3_2_1-00-1.19.hdf5',6,5,2)
    #y_pred5 = main('sj_12_4_3_2_1-38-10.52.hdf5','iq_12_4_3_2_1-55-0.93.hdf5',4,12,1)   
    #y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4)/4
    y_pred = (y_pred1 + y_pred2)/2
    """
    output = open("2221.csv", 'w')
    output.write('city,year,weekofyear,total_cases\n')
    for i in range(len(y_pred)):
        line = ''
        for j in range(3):
            line += str(test_tags[i,j]) + ','
        tmp = abs(round(float(y_pred[i])))
        line += str(int(tmp)) + '\n'
        output.write(line)
    output.close()
    

    #main(model_sj_path, model_iq_path, add,weekConcat,weekPredAfter)
    #print('Elapse time:', time.time()-start_time, 'seconds\n')
    
    