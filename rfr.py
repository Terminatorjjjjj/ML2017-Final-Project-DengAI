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

train_feature_path = sys.argv[1]
train_label_path = sys.argv[2]
test_feature_path = sys.argv[3]
submission_path = sys.argv[4]
prediction_path = sys.argv[5]
run_training = False
run_testing = True

sj_train = []
iq_train = []
predictor = ['week',
			'precipitation_amt_mm',
			'reanalysis_air_temp_k',
			'reanalysis_avg_temp_k',
			'reanalysis_dew_point_temp_k',
			'reanalysis_max_air_temp_k',
			'reanalysis_min_air_temp_k',
			'reanalysis_precip_amt_kg_per_m2',
			'reanalysis_relative_humidity_percent',
			'reanalysis_sat_precip_amt_mm',
			'reanalysis_specific_humidity_g_per_kg',
			'reanalysis_tdtr_k',
			'station_avg_temp_c',
			'station_diur_temp_rng_c',
			'station_max_temp_c',
			'station_min_temp_c',
			'station_precip_mm']
target = 'total_cases'
min_error = np.inf

def preprocess_data(data_path, labels_path=None):
	print('Reading data from:', data_path)
	# load data and set index to city, year, weekofyear
	df = pd.read_csv(data_path, index_col=[0, 1, 2])

	# add back weekofyear
	week = pd.read_csv(data_path, usecols=['weekofyear'])
	df['week'] = pd.Series(week.values.flatten(), index=df.index) 

	# fill missing values
	df.fillna(method='ffill', inplace=True)
	
	# add labels to dataframe
	if labels_path:
		labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
		df = df.join(labels)

	# separate san juan and iquitos
	sj = df.loc['sj']
	iq = df.loc['iq']

	return (sj, iq)

def moving_avg(data, n=3):
	ret = np.cumsum(data)
	ret[n:] = ret[n::] - ret[:-n]
	ret = np.hstack((ret[:n-1],ret[n-1::]/n))
	return ret

def rolling_cross_validation(data, model, cv_split=5, window_size=3):
	global predictor, target
	tscv = TimeSeriesSplit(n_splits=cv_split)
	err = []

	for train_idx, test_idx in tscv.split(data):
		train = data.iloc[train_idx]
		test = data.iloc[test_idx]

		model.fit(train[predictor], train[target])
		result = model.predict(test[predictor])
		result = moving_avg(result, n=window_size)

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
	return rolling_cross_validation(data, model)

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
	param_space = {'max_depth': hp.quniform('max_depth', 1, 200, 1),
					'max_features': hp.quniform('max_features', 1, len(predictor), 1),
					'n_estimators': hp.quniform('n_estimators', 250, 750, 1),
					'min_samples_split': hp.quniform('min_samples_split', 2, 100, 1),
					'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)}

	best_params = fmin(hyperopt_objective, param_space, algo=tpe.suggest, trials=trials, max_evals=1000)

	print('\nBest parameters:', best_params, '\n')
	return best_params

def training(data_path, labels_path):
	print('Start training...')

	### Read training data
	global sj_train, iq_train
	sj_train, iq_train = preprocess_data(data_path, labels_path=labels_path)

	### Training
	global min_error, TRAIN_CITY
	TRAIN_CITY = 'sj'
	sj_trials = Trials()
	sj_params = optimize_model(sj_trials)
	sj_params = {arg: int(value) for arg, value in sj_params.items() if type(value)==type(float(0))}

	TRAIN_CITY = 'iq'
	min_error = np.inf
	iq_trials = Trials()
	iq_params = optimize_model(iq_trials)
	iq_params = {arg: int(value) for arg, value in iq_params.items() if type(value)==type(float(0))}

	### Get best model
	sj_model = RandomForestRegressor(**sj_params)
	sj_model.fit(sj_train[predictor], sj_train[target])
	iq_model = RandomForestRegressor(**iq_params)
	iq_model.fit(iq_train[predictor], iq_train[target])

	with open('sj_rfr_model.pickle','wb') as f:
		pickle.dump(sj_model, f)
	with open('iq_rfr_model.pickle','wb') as f:
		pickle.dump(iq_model, f)

	print('Training is done.')

def testing(data_path, sub_path, pred_path, sj_model_path, iq_model_path):
	print('Start testing...')

	global predictor
	sj_test, iq_test = preprocess_data(data_path)

	with open(sj_model_path,'rb') as f:
		sj_model = pickle.load(f)
	with open(iq_model_path,'rb') as f:
		iq_model = pickle.load(f)

	sj_predictions = sj_model.predict(sj_test[predictor]).astype(int)
	iq_predictions = iq_model.predict(iq_test[predictor]).astype(int)

	submission = pd.read_csv(sub_path, index_col=[0, 1, 2])

	submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
	submission.to_csv(pred_path)

	print('Testing is done.')

def main():

	### Training
	if run_training:
		training(train_feature_path, train_label_path)

	### Testing
	if run_testing:
		sj_model_path = 'sj_rfr_model.pickle'
		iq_model_path = 'iq_rfr_model.pickle'
		testing(test_feature_path, submission_path, prediction_path, sj_model_path, iq_model_path)

if __name__=='__main__':
	start_time = time.time()
	main()
	print('Elapse time:', time.time()-start_time, 'seconds\n')
