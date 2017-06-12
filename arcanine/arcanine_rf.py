import numpy as np
import pandas as pd
from sklearn import dummy, metrics, cross_validation, ensemble
from sklearn.ensemble import RandomForestRegressor
import math

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
"""
# Add new feature
sj_train_feature = sj_train[:,4:]
sj_test_feature = sj_test[:,4:]
iq_train_feature = iq_train[:,4:]
iq_test_feature = iq_test[:,4:]

for feature in add_features:
	sj_train_feature = np.hstack((sj_train_feature,sj_train_dict[feature]**3))

for feature in add_features:
	sj_test_feature = np.hstack((sj_test_feature,sj_test_dict[feature]**3))
	
for feature in add_features:
	iq_train_feature = np.hstack((iq_train_feature,iq_train_dict[feature]**3))
	
for feature in add_features:
	iq_test_feature = np.hstack((iq_test_feature,iq_test_dict[feature]**3))
"""	
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
	
sj_train_feature = sj_train_tmp
iq_train_feature = iq_train_tmp
	
# Normalize
sj_train_mean = np.mean(sj_train_feature, axis=0)
sj_train_std = np.std(sj_train_feature.astype('float'), axis=0)
sj_train_feature = (sj_train_feature - sj_train_mean) / sj_train_std

iq_train_mean = np.mean(iq_train_feature, axis=0)
iq_train_std = np.std(iq_train_feature.astype('float'), axis=0)
iq_train_feature = (iq_train_feature - iq_train_mean) / iq_train_std

# Split into train, val dataset
sj_train, sj_val, sj_train_label, sj_val_label = cross_validation.train_test_split(sj_train_feature, sj_label[:,3][10:])
iq_train, iq_val, iq_train_label, iq_val_label = cross_validation.train_test_split(iq_train_feature, iq_label[:,3][10:]) 

# Random Forest Regression
regressor_sj = RandomForestRegressor(n_estimators=250, min_samples_split=2)
regressor_sj.fit(sj_train, sj_train_label)
regressor_iq = RandomForestRegressor(n_estimators=200, min_samples_split=2)
regressor_iq.fit(iq_train, iq_train_label)

# Evaluate
sj_train_result = regressor_sj.predict(sj_train)
sj_val_result = regressor_sj.predict(sj_val)

for i,j in enumerate(sj_train_result):
	sj_train_result[i] = int(round(j))
for i,j in enumerate(sj_val_result):
	sj_val_result[i] = int(round(j))
	
loss_train = metrics.mean_absolute_error(sj_train_result, sj_train_label)
loss_val = metrics.mean_absolute_error(sj_val_result, sj_val_label)
print('sj Training loss = ' + str(loss_train))
print('sj Validation loss = ' + str(loss_val))

iq_train_result = regressor_iq.predict(iq_train)
iq_val_result = regressor_iq.predict(iq_val)

for i,j in enumerate(iq_train_result):
	iq_train_result[i] = int(round(j))
for i,j in enumerate(iq_val_result):
	iq_val_result[i] = int(round(j))
	
loss_train = metrics.mean_absolute_error(iq_train_result, iq_train_label)
loss_val = metrics.mean_absolute_error(iq_val_result, iq_val_label)
print('iq Training loss = ' + str(loss_train))
print('iq Validation loss = ' + str(loss_val))
"""
# Predict & Output
sj_test_feature = (sj_test_feature - sj_train_mean) / sj_train_std
iq_test_feature = (iq_test_feature - iq_train_mean) / iq_train_std

pred_sj = regressor_sj.predict(sj_test_feature)
pred_iq = regressor_iq.predict(iq_test_feature)

file_write = open('arcanine_prediction_rf.csv', 'w')
file_write.write('city,year,weekofyear,total_cases\n')
for i,row in enumerate(pred_sj):
	string = ''
	for j in sj_test[i][:3]:
		string += (str(j) + ',')
	file_write.write(string + str(int(round(row))) + '\n')
for i,row in enumerate(pred_iq):
	string = ''
	for j in iq_test[i][:3]:
		string += (str(j) + ',')
	file_write.write(string + str(int(round(row))) + '\n')
"""