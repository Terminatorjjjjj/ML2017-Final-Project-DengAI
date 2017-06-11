#!/usr/bin/env python3
#coding=utf-8
import numpy as np
import pandas as pd
import sys
import csv
import time

import h5py
import matplotlib.pyplot as plt
import pickle
import xgboost as xgb


arg = True
loop = 200
if arg == True:
    add = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    eta = float(sys.argv[3])  
    dp = float(sys.argv[4])
    lb = float(sys.argv[5])
else:
    add = 1
    max_depth = 150
    eta = 0.1
    dp = 0.2
    lb = 1
    
log = str(add)+'_'+str(max_depth)+'_'+str(eta)+'_'+str(dp)+'_'+str(lb)

train_feature_file = 'train_feature.csv'
train_label_file = 'train_label.csv'
test_feature_file = 'test_feature.csv'

train_feature = []
train_label = []
test_feature = []

def print_history(h):
	plt.plot(h.history['loss'])
	plt.plot(h.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	fig = plt.gcf()
	fig.savefig('procedure.png') 

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]


    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

def loadFile(city):
    #-----Read training data---------------------------------------------------------
    train_feature = pd.read_csv(train_feature_file, encoding='big5')
    train_feature = train_feature.values
    train_feature = train_feature.astype(str)
    train_feature[train_feature == 'nan'] = '0.0'
    train_feature = np.hstack((train_feature[:,2].reshape(len(train_feature),1), train_feature[:,4::]))
    train_feature = train_feature.astype(float)   
      
    #-----Read testing data----------------------------------------------------------
    test_feature = pd.read_csv(test_feature_file, encoding='big5')
    test_feature = test_feature.values
    test_feature = test_feature.astype(str)
    test_tags = np.asarray(test_feature[:,:3])
    test_feature[test_feature == 'nan'] = '0.0'
    test_feature = np.hstack((test_feature[:,2].reshape(len(test_feature),1), test_feature[:,4::]))
    test_feature = test_feature.astype(float)
    
    # add feature----------------------   
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
        
    
    #-----Read training label--------------------------------------------------------
    train_label = pd.read_csv(train_label_file, encoding='big5')
    train_label = train_label.values
    train_label = train_label[:,3].reshape(len(train_label),1).astype(float)
    
    ### pick out a specific city
    if city == 'sj':
        train_feature = train_feature[:936]
        train_label = train_label[:936]
        test_feature = test_feature[:260]
    
    if city == 'iq':
        train_feature = train_feature[936::]
        train_label = train_label[936::]
        test_feature = test_feature[260::]
    
    return train_feature, train_label, test_feature, test_tags

def normailizeandSaveP(train_feature, train_label, test_feature):
    all_feature = np.vstack((train_feature, test_feature))
    all_mean = np.mean(all_feature, axis=0)
    all_std = np.std(all_feature, axis=0)
    
    train_feature = (train_feature - all_mean) / all_std
    test_feature = (test_feature - all_mean) / all_std
    """
    with open(city + '_' +'deng_mean.pickle','wb') as mm:
        pickle.dump(all_mean, mm)
    with open(city + '_' +'deng_std.pickle','wb') as ss:
        pickle.dump(all_std, ss)    
    """
    return train_feature, train_label, test_feature

def boost(train_feature, train_label, test_feature):
    (x_train,y_train),(x_val,y_val) = split_data(train_feature, train_label, 0.1)
    dtrain = xgb.DMatrix(x_train, y_train)
    dval = xgb.DMatrix(x_val, y_val)
    
    #-----Create model --------------------------------------------------------------
    param = {'max_depth':max_depth, 'eta':eta, 'silent':1, 'objective':'reg:linear',
             'rate_drop':dp, 'lambda':lb,'nthread' :1}
    param['eval_metric'] = 'mae'
    evallist  = [(dtrain,'train'),(dval,'eval')]
    num_round = 1000
    bst = xgb.train( param, dtrain, num_round, evallist,early_stopping_rounds=20)
    bst.save_model(log+'.model')
    dtest = xgb.DMatrix(test_feature)
    ypred = bst.predict(dtest,ntree_limit=bst.best_iteration)
    ypVal = bst.predict(dval,ntree_limit=bst.best_iteration)
    mse = np.sum(np.abs(np.reshape(ypVal,(ypVal.shape[0],1)) - y_val))/(ypVal.shape[0])
    return ypred,bst.best_score, ypVal,mse

t1score = 0
t2score = 0
for i in range(0,loop):
    city = 'sj'
    train_feature, train_label, test_feature, test_tags = loadFile(city)
    train_feature, train_label, test_feature = normailizeandSaveP(train_feature, train_label, test_feature)
    pred1, score1, ypVal1, mse = boost(train_feature, train_label, test_feature)
    pred1 = np.reshape(pred1,(pred1.shape[0],1))
    t1score = t1score + score1
    
    city = 'iq'
    train_feature, train_label, test_feature, test_tags = loadFile(city)
    train_feature, train_label, test_feature = normailizeandSaveP(train_feature, train_label, test_feature)
    pred2, score2, ypVal2, mse2 = boost(train_feature, train_label, test_feature)
    pred2 = np.reshape(pred2,(pred2.shape[0],1))
    t2score = t2score + score2

    print(log + '_' + str(t1score/loop)+  '_' + str(t2score/loop))

    y_pred = np.vstack((pred1, pred2))

    output = open(str(score1)+'_'+str(score2)+'_'+log+'.csv', 'w')
    output.write('city,year,weekofyear,total_cases\n')
    for i in range(len(y_pred)):
        line = ''
        for j in range(3):
            line += str(test_tags[i,j]) + ','
        tmp = abs(round(float(y_pred[i])))
        line += str(int(tmp)) + '\n'
        output.write(line)
    output.close()





