#!/usr/bin/env python3
#coding=utf-8
import numpy as np
import pandas as pd
import sys
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import GRU, LSTM
from keras import regularizers
import matplotlib.pyplot as plt
import keras.backend as K

debugArg = True
if debugArg == True:
    batchSz = int(sys.argv[1])
    epoch = int(sys.argv[2])
    patiences = int(sys.argv[3])
    opt = str(sys.argv[4])
    denseAct = str(sys.argv[5])
    outAct = str(sys.argv[6])    
    hidLayer = int(sys.argv[7])
    city = sys.argv[8]
    lay1 = int(sys.argv[9])
    lay2 = int(sys.argv[10])
    lay3 = int(sys.argv[11])
    lay4 = int(sys.argv[12])
    dDp1 = float(sys.argv[13])
    dDp2 = float(sys.argv[14])
    dDp3 = float(sys.argv[15])
    dDp4 = float(sys.argv[16])
    lay5 = int(sys.argv[17])
    lay6 = int(sys.argv[18])
    lay7 = int(sys.argv[19])
    lay8 = int(sys.argv[20])
    dDp5 = float(sys.argv[21])
    dDp6 = float(sys.argv[22])
    dDp7 = float(sys.argv[23])
    dDp8 = float(sys.argv[24])
    rnnLayer = int(sys.argv[25])
    rnn1U = int(sys.argv[26])
    dpr1 = float(sys.argv[27])
    rnn2U = int(sys.argv[28])
    dpr2 = float(sys.argv[29])
    rnn3U = int(sys.argv[30])
    dpr3 = float(sys.argv[31])
    weekConcat = int(sys.argv[32])
    add = int(sys.argv[33])
    
    
else:
    batchSz = 1
    epoch = 3000
    patiences = 300
    opt = 'adam'
    denseAct = 'relu'
    outAct = 'sigmoid'
    hidLayer = 3
    city = 'iq'
    lay1 = 512
    lay2 = 256
    lay3 = 128
    lay4 = 128
    dDp1 = 0.2
    dDp2 = 0.2
    dDp3 = 0.2
    dDp4 = 0.2
    lay5 = 512
    lay6 = 256
    lay7 = 128
    lay8 = 128
    dDp5 = 0.2
    dDp6 = 0.2
    dDp7 = 0.2
    dDp8 = 0.2
    rnnLayer = 3
    rnn1U = 256
    dpr1 = 0.4
    rnn2U = 128
    dpr2 = 0.4
    rnn3U = 64
    dpr3 = 0.4
    weekConcat = 10
    add = 3
    
if opt != 'adam':
    opt = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
else:
    adam = Adam(lr=0.01, decay=1e-6)
    
log = (str(city)+'_'+ str(weekConcat) + '_' + str(add)+ '_' + str(rnnLayer) + '_'
      + str(hidLayer))
#+ '_' +str(batchSz)+ '_' + str(epoch) + '_' + str(patiences) + '_'  
#      + str(opt)+ '_'  + str(denseAct) + '_'  + str(outAct) + '_' 
#      + str(lay1) + '_'  + str(lay2) + '_'  + str(lay3) + '_'
#      + str(lay4) + str(dDp1)+ '_'  + str(dDp2) + '_'  + str(dDp3) + '_'  
#      + str(dDp4) + str(lay5) + '_'  + str(lay6) + '_'  + str(lay7) + '_'
#      + str(lay8) + str(dDp5)+ '_'  + str(dDp6) + '_'  + str(dDp7) + '_'  
#      + str(dDp8) + '_'  + str(rnn1U) + '_'  + str(dpr1) + '_' 
#      + str(rnn2U) + str(dpr2)+ '_'  + str(rnn3U) + '_'  + str(dpr3)

train_feature_file = 'train_feature.csv'
train_label_file = 'train_label.csv'
test_feature_file = 'test_feature.csv'
save_model_file = 'label'+ city + log + '_' + '.h5py'

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
    #np.random.shuffle(indices)   
    X_data = X[indices]
    Y_data = Y[indices]    
    num_validation_sample = int(split_ratio * X_data.shape[0] )  
    X_train = X_data[num_validation_sample:]    
    Y_train = Y_data[num_validation_sample:]
    X_val = X_data[:num_validation_sample]  
    Y_val = Y_data[:num_validation_sample]
    return (X_train,Y_train),(X_val,Y_val)

def reshape(train_feature, train_label):
    trainRow = train_feature.shape[0]-(weekConcat)
    col = train_feature.shape[1]
    trainF = np.zeros((trainRow, weekConcat, col)) 
    trainL = np.zeros((trainRow,1))
    for i in range(0,trainRow):
        trainF[i] =  train_feature[i:i+weekConcat,:]
        trainL[i] = train_label[i+(weekConcat)] 
    return trainF,  trainL
        

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

#-----Determine different ways to extract feature
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

#-----Join train & test into train_feature
train_feature = np.hstack((train_feature, train_label))


#-----Pick out a specific city
if city == 'sj':
    train_feature = train_feature[:936]
    train_label = train_label[:936]
    test_feature = test_feature[:260]

if city == 'iq':
    train_feature = train_feature[936::]
    train_label = train_label[936::]
    test_feature = test_feature[260::]
    

#-----Normalization
all_feature = np.vstack((train_feature[:,:train_feature.shape[1]-1], test_feature))
all_mean = np.mean(all_feature, axis=0)
all_std = np.std(all_feature, axis=0)
all_mean = np.append(all_mean,np.mean(train_feature[:,train_feature.shape[1]-1], axis=0))
all_std = np.append(all_std,np.std(train_feature[:,train_feature.shape[1]-1], axis=0))
train_feature = (train_feature - all_mean) / all_std


#-----Reshape ---------------------------------------------------------------
train_feature, train_label = reshape(train_feature, train_label)

#-----Create model --------------------------------------------------------------
model = Sequential()
if rnnLayer >= 1:
    if rnnLayer == 1:
        rSeq1 = False
    else: 
        rSeq1 = True
    model.add(LSTM(rnn1U, dropout=dpr1, recurrent_dropout=dpr1, return_sequences=rSeq1,activation='tanh',input_shape=(train_feature.shape[1],train_feature.shape[2],)))

if rnnLayer >=2:
    if rnnLayer == 2:
        rSeq2 = False
    else: 
        rSeq2 = True
    model.add(LSTM(rnn2U, dropout=dpr2, recurrent_dropout=dpr2, return_sequences=rSeq2,activation='tanh'))

if rnnLayer >=3:
    if rnnLayer == 3:
        rSeq3 = False
    else: 
        rSeq3 = True
    model.add(LSTM(rnn3U, dropout=dpr3, recurrent_dropout=dpr3, return_sequences=rSeq3,activation='tanh'))   

model.add(Dense(lay1, activation=denseAct))
model.add(Dropout(dDp1))
if hidLayer >= 2:
    model.add(Dense(lay2, activation=denseAct))
    model.add(Dropout(dDp2))
if hidLayer >= 3:
    model.add(Dense(lay3, activation=denseAct))
    model.add(Dropout(dDp3))
if hidLayer >= 4:
    model.add(Dense(lay4, activation=denseAct))
    model.add(Dropout(dDp4))
if hidLayer >= 5:
    model.add(Dense(lay5, activation=denseAct))
    model.add(Dropout(dDp5))
if hidLayer >= 6:
    model.add(Dense(lay6, activation=denseAct))
    model.add(Dropout(dDp6))
if hidLayer >= 7:
    model.add(Dense(lay7, activation=denseAct))
    model.add(Dropout(dDp7))
if hidLayer >= 8:
    model.add(Dense(lay8, activation=denseAct))
    model.add(Dropout(dDp8))
model.add(Dense(1,activation='linear'))
model.summary()

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

history = History()
es = EarlyStopping(monitor='val_loss', patience=patiences, verbose=1, mode='min')
ck = ModelCheckpoint(log+'-{epoch:02d}-{val_loss:.2f}.hdf5', 
                     verbose=1,
                     save_best_only=True,
                     save_weights_only=False,
                     monitor='val_loss',
                     mode='min')

(x_train,y_train),(x_val,y_val) = split_data(train_feature, train_label, 0.1)

h = model.fit(x_train, y_train, epochs=epoch, batch_size=batchSz,
                 validation_data=(x_val, y_val), 
                 callbacks=[es, ck, history])



trainEpoch = len(h.history['val_loss'])
print(city+ ' model min loss: ', min(h.history['val_loss']))
with open (str(trainEpoch) + '_' + city + str(min(h.history['val_loss'])) + '_' +log + '.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow('a')



