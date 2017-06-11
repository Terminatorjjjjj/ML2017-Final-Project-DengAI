import numpy as np
import pandas as pd
import math
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import regularizers
from sklearn import dummy, metrics, cross_validation, ensemble

features = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
       'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
       'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
       'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
	   'station_min_temp_c', 'station_precip_mm']

add_features = ['reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k', 
	'reanalysis_specific_humidity_g_per_kg', 'station_min_temp_c',
	'station_avg_temp_c']

def DNN_sj(x_train, y_train, x_val, y_val):
	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1, activation='linear'))
	model.summary()
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

	es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')
	ck = ModelCheckpoint(filepath='arcanine_sj.h5', verbose=1, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')

	h = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_data=(x_val, y_val), callbacks=[es, ck])
	return model

def DNN_iq(x_train, y_train, x_val, y_val):
	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.7))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1, activation='linear'))
	model.summary()
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

	es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min')
	ck = ModelCheckpoint(filepath='arcanine_iq.h5', verbose=1, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min')

	h = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_data=(x_val, y_val), callbacks=[es, ck])
	return model

# Read validation & testing data
df_train = pd.read_csv('train_feature.csv',encoding="big5")
df_train.fillna(method='ffill', inplace=True)
train = df_train.as_matrix()

df_train_label = pd.read_csv('train_label.csv',encoding="big5")
label = df_train_label.as_matrix()

df_test = pd.read_csv('test_feature.csv',encoding="big5")
df_test.fillna(method='ffill', inplace=True)
test = df_test.as_matrix()

# Construct vector dictionary
sj_train = train[:936]
iq_train = train[936:]
sj_test = test[:260]
iq_test = test[260:]
sj_label = label[:936]
iq_label = label[936:]

file_read = open('train_feature.csv', 'r')
line = file_read.readline()
line = line.split(',')
line = line[4:]
sj_train_dict = {}
sj_test_dict = {}
iq_train_dict = {}
iq_test_dict = {}

for i,feature in enumerate(line):
	sj_train_dict[feature] = sj_train[:,i+4].reshape((len(sj_train[:,i+4]),1))

for i,feature in enumerate(line):
	sj_test_dict[feature] = sj_test[:,i+4].reshape((len(sj_test[:,i+4]),1))

for i,feature in enumerate(line):
	iq_train_dict[feature] = iq_train[:,i+4].reshape((len(iq_train[:,i+4]),1))

for i,feature in enumerate(line):
	iq_test_dict[feature] = iq_test[:,i+4].reshape((len(iq_test[:,i+4]),1))

# Add new feature
sj_train_feature = sj_train[:,4:]
sj_test_feature = sj_test[:,4:]
iq_train_feature = iq_train[:,4:]
iq_test_feature = iq_test[:,4:]

for feature in add_features:
	sj_train_feature = np.hstack((sj_train_feature,sj_train_dict[feature]**2))

for feature in add_features:
	sj_test_feature = np.hstack((sj_test_feature,sj_test_dict[feature]**2))
	
for feature in add_features:
	iq_train_feature = np.hstack((iq_train_feature,iq_train_dict[feature]**2))
	
for feature in add_features:
	iq_test_feature = np.hstack((iq_test_feature,iq_test_dict[feature]**2))

# Normalize
mean = np.mean(sj_train_feature, axis=0)
std = np.std(sj_train_feature.astype('float'), axis=0)
sj_train_feature = (sj_train_feature - mean) / std

mean = np.mean(sj_test_feature, axis=0)
std = np.std(sj_test_feature.astype('float'), axis=0)
sj_test_feature = (sj_test_feature - mean) / std

mean = np.mean(iq_train_feature, axis=0)
std = np.std(iq_train_feature.astype('float'), axis=0)
iq_train_feature = (iq_train_feature - mean) / std

mean = np.mean(iq_test_feature, axis=0)
std = np.std(iq_test_feature.astype('float'), axis=0)
iq_test_feature = (iq_test_feature - mean) / std

# Split into train, val dataset
sj_train, sj_val, sj_train_label, sj_val_label = cross_validation.train_test_split(sj_train_feature, sj_label[:,3])
iq_train, iq_val, iq_train_label, iq_val_label = cross_validation.train_test_split(iq_train_feature, iq_label[:,3]) 

# DNN
#model = DNN_sj(sj_train, sj_train_label, sj_val, sj_val_label)
#model_1 = DNN_iq(iq_train, iq_train_label, iq_val, iq_val_label)

# load model
model = load_model('arcanine_sj.h5')
model_1 = load_model('arcanine_iq.h5')
 
# Evaluate
pred = model_1.predict(iq_val)
loss = metrics.mean_absolute_error(pred.astype('int'), iq_val_label)

# Predict & Output
pred_sj = model.predict(sj_test_feature)
pred_iq = model_1.predict(iq_test_feature)

file_write = open('arcanine_prediction.csv', 'w')
file_write.write('city,year,weekofyear,total_cases\n')
for i,row in enumerate(pred_sj):
	string = ''
	for j in sj_test[i][:3]:
		string += (str(j) + ',')
	file_write.write(string + str(int(math.ceil(row))) + '\n')
for i,row in enumerate(pred_iq):
	string = ''
	for j in iq_test[i][:3]:
		string += (str(j) + ',')
	file_write.write(string + str(int(math.ceil(row))) + '\n')