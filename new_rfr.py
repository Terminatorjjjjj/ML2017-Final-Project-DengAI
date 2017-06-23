#!/usr/bin/env python3
#coding=utf-8

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pickle
import matplotlib.pyplot as plt

train_feature_path = './data/train_feature.csv'
train_label_path = './data/train_label.csv'
test_feature_path = './data/test_feature.csv'
submission_path = './data/submission_format.csv'
prediction_path = 'pred_rfr_new.csv'

sj_train_model = './new_rfr_models/sj_rfr.pickle'
iq_train_model = './new_rfr_models/iq_rfr.pickle'
sj_test_model = './new_rfr_models/sj_rfr.pickle'
iq_test_model = './new_rfr_models/iq_rfr.pickle'

run_training = True
run_testing = True
MAX_EVAL = 1500

NUM_WINDOW = 3
NUM_LAG_FEATURE = 3
NUM_LAG_LABEL = 4
NUM_SHIFT_WEEK = 1
NUM_MERGE_WEEKS = 20

sj_train = []
iq_train = []
predictors = ['reanalysis_specific_humidity_g_per_kg', 
			'reanalysis_dew_point_temp_k', 
			'station_avg_temp_c', 
			'station_min_temp_c',
			'week']
target = 'total_cases'
min_error = np.inf
pred = []

def read_data(data_path, labels_path=None):
	print('Reading data from:', data_path)
	# load data and set index to city, year, weekofyear
	df = pd.read_csv(data_path, index_col=[0, 1, 2])

	# add back weekofyear
	week = pd.read_csv(data_path, usecols=['weekofyear']).values
	cosine_week = np.cos(2*np.pi*(week/np.max(week)))
	df['week'] = pd.Series(cosine_week.flatten(), index=df.index) 

	# fill missing values
	df.fillna(method='ffill', inplace=True)
	
	# add labels to dataframe
	if labels_path:
		labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
		df = df.join(labels)
	else:
		dummy_label = np.zeros((df.shape[0],1))
		df[target] = pd.Series(dummy_label.flatten(), index=df.index)
	
	# separate san juan and iquitos
	sj = df.loc['sj']
	iq = df.loc['iq']

	return (sj, iq)

def plot_result(title, line1, line2=None):
	print('Plot:', title)
	plt.plot(line1)
	if line2:
		plt.plot(line2)
	plt.title(title)
	plt.xlabel('Weeks')
	plt.ylabel('Total Cases')
	if line2:
		plt.legend(['Prediction', 'True'], loc='upper left')
	plt.savefig(title + '.png')

def merge_feature(sj_train, iq_train, sj_test, iq_test):
	ret_sj_test = pd.concat(objs=[sj_train[-1*NUM_MERGE_WEEKS::], sj_test])
	ret_iq_test = pd.concat(objs=[iq_train[-1*NUM_MERGE_WEEKS::], iq_test])
	return ret_sj_test, ret_iq_test

def preprocessing(df, predictors, num_shift_week, num_lag_feature, num_lag_label):
	ret_df = df.copy()
	ret_pred = list(predictors)

	if num_shift_week:
		print('Data shift week:', num_shift_week)
		ret_df[target] = ret_df[target].shift(-1*num_shift_week)
		ret_df.fillna(method='ffill', inplace=True)

	if num_lag_feature:
		print('Data lag feature week:', num_lag_feature)
		for feature in predictors:
			for i in range(num_lag_feature):
				lagging_feature = feature + '_lag_' + str(i+1)
				ret_df[lagging_feature] = ret_df[feature].shift(i+1)
				if lagging_feature not in ret_pred:
					ret_pred.append(lagging_feature)
		ret_df.fillna(method='bfill', inplace=True)

	if num_lag_label:
		print('Data lag label week:', num_lag_label)
		for i in range(num_lag_label):
			lagging_label = target + '_lag_' + str(i+1)
			ret_df[lagging_label] = ret_df[target].shift(i+1)
			if lagging_label not in ret_pred:
				ret_pred.append(lagging_label)
		ret_df.fillna(method='bfill', inplace=True)

	return ret_df, ret_pred

def moving_avg(data, n=3):
	ret = np.cumsum(data).reshape(len(data),1)
	ret[n::] = ret[n::] - ret[:-n]
	ret = np.vstack((data[:n-1],ret[n-1::]/n))
	return ret

def rolling_cross_validation(data, model, cv_split=5, window_size=3):
	tscv = TimeSeriesSplit(n_splits=cv_split)
	err = []

	for train_idx, test_idx in tscv.split(data):
		train = data.iloc[train_idx]
		test = data.iloc[test_idx]

		model.fit(train[pred], train[target])
		num_data = test.shape[0]
		result = np.zeros((num_data,1))
		for i in range(num_data):
			data = test[pred].copy()
			result[i] = model.predict(data.iloc[i,:].values.reshape(1,-1)).astype(float)
			for j in range(NUM_LAG_LABEL):
				if i+j < num_data:
					lagging_label = target + '_lag_' + str(j+1)
					col = test.columns.get_loc(lagging_label)
					test.iloc[i+j,col] = result[i]

		e = mean_absolute_error(np.round(result), test[target])
		err.append(e)

	return np.mean(err)

def hyperopt_training(params):
	model = RandomForestRegressor(**params)

	global sj_train, iq_train, TRAIN_CITY
	if TRAIN_CITY == 'sj':
		data = sj_train
	else:
		data = iq_train
	return rolling_cross_validation(data, model, window_size=NUM_WINDOW)

def hyperopt_objective(params):
	global min_error

	params = {arg: int(value) for arg, value in params.items() if type(value)==type(float(0))}
	
	e = hyperopt_training(params)
	if e < min_error:
		min_error = e
		print('New best:', e, 'params:', params)

	return {'loss': e, 'status': STATUS_OK}

def optimize_model(trials):
	print('Optimizing model for city:', TRAIN_CITY)
	param_space = {'max_depth': hp.quniform('max_depth', 1, 200, 1), # 1-200
					'max_features': hp.quniform('max_features', 1, len(pred), 1),
					'n_estimators': hp.quniform('n_estimators', 250, 750, 1), # 250-750
					'min_samples_split': hp.quniform('min_samples_split', 2, 100, 1),
					'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 100, 1)}

	best_params = fmin(hyperopt_objective, param_space, algo=tpe.suggest, trials=trials, max_evals=MAX_EVAL)
	return best_params

def get_best_model(sj_params, iq_params):
	print('Train best models.')
	sj_model = RandomForestRegressor(**sj_params)
	sj_model.fit(sj_train[pred], sj_train[target])
	iq_model = RandomForestRegressor(**iq_params)
	iq_model.fit(iq_train[pred], iq_train[target])

	with open(sj_train_model,'wb') as f:
		pickle.dump(sj_model, f)
	with open(iq_train_model,'wb') as f:
		pickle.dump(iq_model, f)

def training(data_path, labels_path):
	print('Start training...')

	### Read training data
	global sj_train, iq_train
	sj_train, iq_train = read_data(data_path, labels_path=labels_path)

	### Preprocessing
	global pred
	sj_train, pred = preprocessing(sj_train, predictors, NUM_SHIFT_WEEK, NUM_LAG_FEATURE, NUM_LAG_LABEL)
	iq_train, pred = preprocessing(iq_train, predictors, NUM_SHIFT_WEEK, NUM_LAG_FEATURE, NUM_LAG_LABEL)

	### Training
	global min_error, TRAIN_CITY
	TRAIN_CITY = 'sj'
	sj_trials = Trials()
	sj_params = optimize_model(sj_trials)
	sj_params = {arg: int(value) for arg, value in sj_params.items() if type(value)==type(float(0))}
	sj_best = min_error

	TRAIN_CITY = 'iq'
	min_error = np.inf
	iq_trials = Trials()
	iq_params = optimize_model(iq_trials)
	iq_params = {arg: int(value) for arg, value in iq_params.items() if type(value)==type(float(0))}
	iq_best = min_error

	### Print training results
	print('\nData shift week:', NUM_SHIFT_WEEK)
	print('Data lag feature week:', NUM_LAG_FEATURE)
	print('Data lag label week:', NUM_LAG_LABEL)
	print('Predictors:', pred)
	print('sj:', sj_best, ',', sj_params, '\n')
	print('iq:', iq_best, ',', iq_params, '\n')
	print('MAE:', (sj_best*936 + iq_best*520) / 1456, '\n')

	### Get best model
	get_best_model(sj_params, iq_params)

	print('Training is done.')

def make_prediction(data, model, num_lag_label):
	num_data = data.shape[0]
	result = np.zeros((num_data,1)).astype(float)

	for i in range(num_data):
		test = data[pred].copy()
		result[i] = model.predict(test.iloc[i,:].values.reshape(1,-1)).astype(float)
		for j in range(num_lag_label):
			if i+j < num_data:
				lagging_label = target + '_lag_' + str(j+1)
				col = data.columns.get_loc(lagging_label)
				data.iloc[i+j,col] = result[i]

	return result

def make_submission(result, id_path=test_feature_path, pred_path=prediction_path):
	pred_index = pd.read_csv(id_path, index_col=[0, 1, 2])
	prediction = pd.DataFrame(index=pred_index.index.copy())
	prediction[target] = result
	prediction.to_csv(pred_path)
	print('Save predictions to', pred_path)

def testing(train_path, labels_path, test_path, pred_path, sj_model_path, iq_model_path):
	print('Start testing...')

	sj_train, iq_train = read_data(train_path, labels_path=labels_path)
	sj_test, iq_test = read_data(test_path)
	sj_test, iq_test = merge_feature(sj_train, iq_train, sj_test, iq_test)

	### Preprocessing
	global pred
	sj_test, pred = preprocessing(sj_test, predictors, NUM_SHIFT_WEEK, NUM_LAG_FEATURE, NUM_LAG_LABEL)
	iq_test, pred = preprocessing(iq_test, predictors, NUM_SHIFT_WEEK, NUM_LAG_FEATURE, NUM_LAG_LABEL)

	print('Predictors:', pred)

	### Remove training data part
	sj_test = sj_test[NUM_MERGE_WEEKS::]
	iq_test = iq_test[NUM_MERGE_WEEKS::]

	with open(sj_model_path,'rb') as f:
		sj_model = pickle.load(f)
	print('Loading sj model from: ', sj_model_path)
	with open(iq_model_path,'rb') as f:
		iq_model = pickle.load(f)
	print('Loading iq model from: ', iq_model_path)

	sj_result = make_prediction(sj_test, sj_model, NUM_LAG_LABEL)
	iq_result = make_prediction(iq_test, iq_model, NUM_LAG_LABEL)

	print('Moving average window size:', NUM_WINDOW)
	sj_result = np.round(moving_avg(sj_result, n=NUM_WINDOW)).astype(int)
	iq_result = np.round(moving_avg(iq_result, n=NUM_WINDOW)).astype(int)
	
	make_submission(np.concatenate([sj_result, iq_result]))

	plot_result('Testing', np.concatenate([sj_result, iq_result]))

	print('Testing is done.')

def main():

	### Training
	if run_training:
		training(train_feature_path, train_label_path)

	### Testing
	if run_testing:
		testing(train_feature_path, train_label_path, test_feature_path, prediction_path, sj_test_model, iq_test_model)

if __name__=='__main__':
	start_time = time.time()
	main()
	print('Elapse time:', time.time()-start_time, 'seconds\n')
