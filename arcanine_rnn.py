# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import dummy, metrics, cross_validation, ensemble

def rnn_sj(x_train, y_train, x_val, y_val):
	model = Sequential()
	model.add(GRU(256, dropout=0.2, recurrent_dropout=0.2, input_shape=(x_train.shape[1], x_train.shape[2],)))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='linear'))
	model.summary()
	
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

	es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
	ck = ModelCheckpoint(filepath='arcanine_rnn_sj.h5', verbose=1, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')
	
	h = model.fit(x_train, y_train, epochs=300, batch_size=128, validation_data=(x_val, y_val), callbacks=[es, ck])
	return model
	
# Read training & validation & testing data
df_train = pd.read_csv('train_feature.csv',encoding="big5")
df_train.fillna(method='ffill', inplace=True)
train = df_train.as_matrix()

df_train_label = pd.read_csv('train_label.csv',encoding="big5")
label = df_train_label.as_matrix()

df_test = pd.read_csv('test_feature.csv',encoding="big5")
df_test.fillna(method='ffill', inplace=True)
test = df_test.as_matrix()

# Split sj & iq
sj_train = train[:936]
iq_train = train[936:]
sj_test = test[:260]
iq_test = test[260:]
sj_label = label[:936]
iq_label = label[936:]

sj_train_feature = sj_train[:,4:]
sj_test_feature = sj_test[:,4:]
iq_train_feature = iq_train[:,4:]
iq_test_feature = iq_test[:,4:]

# Normalize
sj_train_mean = np.mean(sj_train_feature, axis=0)
sj_train_std = np.std(sj_train_feature.astype('float'), axis=0)
sj_train_feature = (sj_train_feature - sj_train_mean) / sj_train_std

iq_train_mean = np.mean(iq_train_feature, axis=0)
iq_train_std = np.std(iq_train_feature.astype('float'), axis=0)
iq_train_feature = (iq_train_feature - iq_train_mean) / iq_train_std

# Split into train, val dataset
sj_train, sj_val, sj_train_label, sj_val_label = cross_validation.train_test_split(sj_train_feature, sj_label[:,3])
iq_train, iq_val, iq_train_label, iq_val_label = cross_validation.train_test_split(iq_train_feature, iq_label[:,3]) 
sj_train = sj_train.reshape(sj_train.shape[0], sj_train.shape[1], 1)
sj_val = sj_val.reshape(sj_val.shape[0], sj_val.shape[1], 1)

# RNN
rnn_sj(sj_train, sj_train_label, sj_val, sj_val_label)

# Load model
model_sj = load_model('arcanine_rnn_sj.h5')

# Evaluate
loss_train = metrics.mean_absolute_error(model_sj.predict(sj_train), sj_train_label)
loss_val = metrics.mean_absolute_error(model_sj.predict(sj_val), sj_val_label)
print('Training loss = ' + str(loss_train))
print('Validation loss = ' + str(loss_val))
