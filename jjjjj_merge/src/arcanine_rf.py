#!/usr/bin/env python3
#coding=utf-8

import numpy as np
import pandas as pd
from sklearn import dummy, metrics, cross_validation, ensemble
from sklearn.ensemble import RandomForestRegressor
import math
import pickle
import time

start_time = time.time()

train_feature_path = '../data/train_feature.csv'
train_label_path = '../data/train_label.csv'
test_feature_path = '../data/test_feature.csv'
prediction_path = '../arc.csv'
merge_path = ''
sj_model_path = '../arc/arc_sj_6-9267-300.pickle'
iq_model_path = '../arc/arc_iq_3-3047-250.pickle'

TRAIN = False
ESTIMATORS = 500

features = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
       'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
       'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
       'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
	   'station_min_temp_c', 'station_precip_mm']

add_features = ['reanalysis_specific_humidity_g_per_kg', 'station_min_temp_c', 'station_avg_temp_c']

# Read training & validation & testing data
print('Reading data from:', train_feature_path)
df_train = pd.read_csv(train_feature_path,encoding="big5")
df_train.fillna(method='ffill', inplace=True)
train = df_train.as_matrix()

print('Reading data from:', train_label_path)
df_train_label = pd.read_csv(train_label_path,encoding="big5")
label = df_train_label.as_matrix()

print('Reading data from:', test_feature_path)
df_test = pd.read_csv(test_feature_path,encoding="big5")
df_test.fillna(method='ffill', inplace=True)
test = df_test.as_matrix()

# Construct vector dictionary
sj_train = train[:936]
iq_train = train[936:]
sj_test = test[:260]
iq_test = test[260:]
sj_label = label[:936]
iq_label = label[936:]

del train, test, label

file_read = open(train_feature_path, 'r')
line = file_read.readline()
line = line.split(',')
line = line[4:]
sj_train_dict = {}
sj_test_dict = {}
iq_train_dict = {}
iq_test_dict = {}

for i,feature in enumerate(line):
	sj_train_dict[feature] = sj_train[:,i+4].reshape((len(sj_train[:,i+4]),1))
	sj_test_dict[feature] = sj_test[:,i+4].reshape((len(sj_test[:,i+4]),1))
	iq_train_dict[feature] = iq_train[:,i+4].reshape((len(iq_train[:,i+4]),1))
	iq_test_dict[feature] = iq_test[:,i+4].reshape((len(iq_test[:,i+4]),1))

# Add new feature
print('Select features.')
sj_train_feature = sj_train[:,4:]
sj_test_feature = sj_test[:,4:]
iq_train_feature = iq_train[:,4:]
iq_test_feature = iq_test[:,4:]

for feature in add_features:
	sj_train_feature = np.hstack((sj_train_feature,sj_train_dict[feature]**3))
	sj_test_feature = np.hstack((sj_test_feature,sj_test_dict[feature]**3))
	iq_train_feature = np.hstack((iq_train_feature,iq_train_dict[feature]**3))
	iq_test_feature = np.hstack((iq_test_feature,iq_test_dict[feature]**3))

# Choose feature
sj_train_feature = sj_train_dict['reanalysis_dew_point_temp_k']
sj_test_feature = sj_test_dict['reanalysis_dew_point_temp_k']
iq_train_feature = iq_train_dict['reanalysis_dew_point_temp_k']
iq_test_feature = iq_test_dict['reanalysis_dew_point_temp_k']

for feature in add_features:
	sj_train_feature = np.hstack((sj_train_feature,sj_train_dict[feature]))
	sj_test_feature = np.hstack((sj_test_feature,sj_test_dict[feature]))
	iq_train_feature = np.hstack((iq_train_feature,iq_train_dict[feature]))
	iq_test_feature = np.hstack((iq_test_feature,iq_test_dict[feature]))

sj_train_feature = np.hstack((sj_train_feature,sj_train_dict['reanalysis_dew_point_temp_k']**3))
sj_test_feature = np.hstack((sj_test_feature,sj_test_dict['reanalysis_dew_point_temp_k']**3))
iq_train_feature = np.hstack((iq_train_feature,iq_train_dict['reanalysis_dew_point_temp_k']**3))
iq_test_feature = np.hstack((iq_test_feature,iq_test_dict['reanalysis_dew_point_temp_k']**3))
	
for feature in add_features:
	sj_train_feature = np.hstack((sj_train_feature,sj_train_dict[feature]**3))
	sj_test_feature = np.hstack((sj_test_feature,sj_test_dict[feature]**3))
	iq_train_feature = np.hstack((iq_train_feature,iq_train_dict[feature]**3))
	iq_test_feature = np.hstack((iq_test_feature,iq_test_dict[feature]**3))

# Add ten weeks feature
sj_train_tmp = np.zeros((len(sj_train_feature)-10,90))
iq_train_tmp = np.zeros((len(iq_train_feature)-10,90))
#sj_train_tmp = sj_train_feature[9:,:]
#iq_train_tmp = iq_train_feature[9:,:]

for i in range(len(sj_train_tmp)):
	tmp = sj_train_feature[i+9]
	for j in range(i+8, i-1, -1):
		tmp = np.hstack((tmp,sj_train_feature[j]))
	tmp = np.hstack((tmp,sj_label[:,3][i:i+10]))
	sj_train_tmp[i] = tmp

for i in range(len(iq_train_tmp)):
	tmp = iq_train_feature[i+9]
	for j in range(i+8, i-1, -1):
		tmp = np.hstack((tmp,iq_train_feature[j]))
	tmp = np.hstack((tmp,iq_label[:,3][i:i+10]))
	iq_train_tmp[i] = tmp
	
sj_train_feature_1 = sj_train_tmp
iq_train_feature_1 = iq_train_tmp
	
# Normalize
print('Normalization.')
sj_train_mean = np.mean(sj_train_feature_1, axis=0)
sj_train_std = np.std(sj_train_feature_1.astype('float'), axis=0)
sj_train_feature_1 = (sj_train_feature_1 - sj_train_mean) / sj_train_std

iq_train_mean = np.mean(iq_train_feature_1, axis=0)
iq_train_std = np.std(iq_train_feature_1.astype('float'), axis=0)
iq_train_feature_1 = (iq_train_feature_1 - iq_train_mean) / iq_train_std

if TRAIN:
	# Split into train, val dataset
	print('Split data.')
	sj_train, sj_val, sj_train_label, sj_val_label = cross_validation.train_test_split(sj_train_feature_1, sj_label[:,3][10:])
	iq_train, iq_val, iq_train_label, iq_val_label = cross_validation.train_test_split(iq_train_feature_1, iq_label[:,3][10:]) 

	# Random Forest Regression
	regressor_sj = RandomForestRegressor(n_estimators=ESTIMATORS, min_samples_split=2)
	regressor_sj.fit(sj_train, sj_train_label)
	regressor_iq = RandomForestRegressor(n_estimators=ESTIMATORS-50, min_samples_split=2)
	regressor_iq.fit(iq_train, iq_train_label)

	# Evaluate
	print('Training sj.')
	sj_train_result = regressor_sj.predict(sj_train)
	sj_val_result = regressor_sj.predict(sj_val)

	for i,j in enumerate(sj_train_result):
		sj_train_result[i] = int(round(j))
	for i,j in enumerate(sj_val_result):
		sj_val_result[i] = int(round(j))
		
	loss_train = metrics.mean_absolute_error(sj_train_result, sj_train_label)
	loss_val = metrics.mean_absolute_error(sj_val_result, sj_val_label)
	loss_val_sj = loss_val
	print('Training loss   = ' + str(loss_train))
	print('Validation loss = ' + str(loss_val))

	print('Training iq.')
	iq_train_result = regressor_iq.predict(iq_train)
	iq_val_result = regressor_iq.predict(iq_val)

	for i,j in enumerate(iq_train_result):
		iq_train_result[i] = int(round(j))
	for i,j in enumerate(iq_val_result):
		iq_val_result[i] = int(round(j))
		
	loss_train = metrics.mean_absolute_error(iq_train_result, iq_train_label)
	loss_val = metrics.mean_absolute_error(iq_val_result, iq_val_label)
	print('Training loss   = ' + str(loss_train))
	print('Validation loss = ' + str(loss_val))

	# Save model
	print('Saving models.')
	with open('arc_sj.pickle', 'wb') as f:
	    pickle.dump(regressor_sj, f)
	with open('arc_iq.pickle', 'wb') as f:
	    pickle.dump(regressor_iq, f)
else:
	# Load model
	print('Load models.')
	with open(sj_model_path, 'rb') as f:
		regressor_sj = pickle.load(f)
	with open(iq_model_path, 'rb') as f:
		regressor_iq = pickle.load(f)

# Predict & Output
pred_sj = np.asarray([0.0 for i in range(len(sj_test_feature))])
pred_iq = np.asarray([0.0 for i in range(len(iq_test_feature))])

print('Testing sj.')
for i in range(len(sj_test_feature)):
	if i == 0:
		tmp = sj_train_feature[len(sj_train_feature)-1]
		for j in range(len(sj_train_feature)-2, len(sj_train_feature)-11, -1):
			tmp = np.hstack((tmp,sj_train_feature[j]))
		reg_feature = tmp
		tmp = np.hstack((tmp,sj_label[:,3][len(sj_train_feature)-10:len(sj_train_feature)]))
		tmp = (tmp - sj_train_mean) / sj_train_std
		pred_sj[i] = regressor_sj.predict(tmp.reshape((1, -1)))
	elif (i > 0) and (i < 10):
		tmp = sj_test_feature[i-1]
		if i > 1:
			for j in range(i-2, -1, -1):
				tmp = np.hstack((tmp,sj_test_feature[j]))
		tmp = np.hstack((tmp,reg_feature[:80-8*i]))
		tmp = np.hstack((tmp,sj_label[:,3][len(sj_train_feature)-10+i:len(sj_train_feature)]))
		for j in range(0, i, 1):
			tmp = np.hstack((tmp,pred_sj[j]))
		tmp = (tmp - sj_train_mean) / sj_train_std
		pred_sj[i] = regressor_sj.predict(tmp.reshape((1, -1)))
	else:
		tmp = sj_test_feature[i-1]
		for j in range(i-2, i-11, -1):
			tmp = np.hstack((tmp,sj_test_feature[j]))
		for j in range(i-10, i, 1):
			tmp = np.hstack((tmp,pred_sj[j]))
		tmp = (tmp - sj_train_mean) / sj_train_std
		pred_sj[i] = regressor_sj.predict(tmp.reshape((1, -1)))

print('Testing iq.')
for i in range(len(iq_test_feature)):
	if i == 0:
		tmp = iq_train_feature[len(iq_train_feature)-1]
		for j in range(len(iq_train_feature)-2, len(iq_train_feature)-11, -1):
			tmp = np.hstack((tmp,iq_train_feature[j]))
		reg_feature = tmp
		tmp = np.hstack((tmp,iq_label[:,3][len(iq_train_feature)-10:len(iq_train_feature)]))
		tmp = (tmp - iq_train_mean) / iq_train_std
		pred_iq[i] = regressor_iq.predict(tmp.reshape((1, -1)))
	elif (i > 0) and (i < 10):
		tmp = iq_test_feature[i-1]
		if i > 1:
			for j in range(i-2, -1, -1):
				tmp = np.hstack((tmp,iq_test_feature[j]))
		tmp = np.hstack((tmp,reg_feature[:80-8*i]))
		tmp = np.hstack((tmp,iq_label[:,3][len(iq_train_feature)-10+i:len(iq_train_feature)]))
		for j in range(0, i, 1):
			tmp = np.hstack((tmp,pred_iq[j]))
		tmp = (tmp - iq_train_mean) / iq_train_std
		pred_iq[i] = regressor_iq.predict(tmp.reshape((1, -1)))
	else:
		tmp = iq_test_feature[i-1]
		for j in range(i-2, i-11, -1):
			tmp = np.hstack((tmp,iq_test_feature[j]))
		for j in range(i-10, i, 1):
			tmp = np.hstack((tmp,pred_iq[j]))
		tmp = (tmp - iq_train_mean) / iq_train_std
		pred_iq[i] = regressor_iq.predict(tmp.reshape((1, -1)))

if merge_path:
	file_read = pd.read_csv(merge_path,encoding="big5")
	yuah = file_read.as_matrix()
	yuah_pred_sj = yuah[:,3][:260]
	yuah_pred_iq = yuah[:,3][260:]
"""
import matplotlib.pyplot as plt
file_read_new = pd.read_csv('arcanine_prediction_rf.csv',encoding="big5")
arcanine = file_read_new.as_matrix()
arcanine_pred_sj = arcanine[:,3][:260]
arcanine_pred_iq = arcanine[:,3][260:]

# Plot the trend
pred_sj = pred_sj*0.2 + yuah_pred_sj*0.8
pred_iq = pred_iq*0.2 + yuah_pred_iq*0.8

fig = plt.figure()
plt.plot(np.arange(0, arcanine_pred_sj.shape[0]), arcanine_pred_sj, 'b')
plt.plot(np.arange(0, pred_sj.shape[0]), pred_sj, 'r')
plt.plot(np.arange(0, yuah_pred_sj.shape[0]), yuah_pred_sj, 'g')
plt.show()
fig.savefig('arcanine.png')
"""
# Endemble result
if merge_path:
	pred_sj = pred_sj*0.2 + yuah_pred_sj*0.8
	pred_iq = pred_iq*0.2 + yuah_pred_iq*0.8

print('Saving prediction to:', prediction_path)
file_write = open(prediction_path, 'w')
file_write.write('city,year,weekofyear,total_cases\n')
for i,row in enumerate(pred_sj):
	string = ''
	for j in sj_test[i][:3]:
		string += (str(j) + ',')
	file_write.write(string + str(row) + '\n')
for i,row in enumerate(pred_iq):
	string = ''
	for j in iq_test[i][:3]:
		string += (str(j) + ',')
	file_write.write(string + str(row) + '\n')

print('Elapse time:', time.time()-start_time, 'seconds\n')
