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
WINDOW = 3
MAX_EVAL = 1500

ADD_LAGGING = True
LAGGING_WEEK = 9
ADD_FULL_LAG = True
EXPWA = False

SHIFT_WEEK = False
SHIFT_NUM = 1

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
core = ['reanalysis_specific_humidity_g_per_kg', 
		'reanalysis_dew_point_temp_k', 
		'station_avg_temp_c', 
		'station_min_temp_c',
		'week']
target = 'total_cases'
min_error = np.inf
lag_predictor = []
pred = []
predictor = core
print('Core as pred.')

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

def set_predictor(lagging=ADD_LAGGING):
	global pred
	if lagging:
		pred = lag_predictor
		print('Set lagging predictors.')
	else:
		pred = predictor
		print('Set normal predictors.')

def add_lagging_data(df, lag_num=LAGGING_WEEK, full_data=ADD_FULL_LAG, expwa=EXPWA):
	global lag_predictor
	lag_predictor = list(predictor)
	# decide lagging data tp be added
	if ADD_FULL_LAG:
		features = list(predictor)
		print('Add full lagging data:', LAGGING_WEEK, 'week(s)')
	else:
		features = list(core)
		print('Add core lagging data:', LAGGING_WEEK, 'week(s)')

	# add lagging data
	for feature in features:
		for i in range(lag_num):
			lagging_feature = feature + '_lag' + str(i+1)
			df[lagging_feature] = df[feature].shift(i+1)
			if lagging_feature not in lag_predictor:
				lag_predictor.append(lagging_feature)

	# fill nan in first lag_num rows (not having lagging data)
	df.fillna(method='bfill', inplace=True)
	# df.fillna(df.mean(), inplace=True)

	return df

def shift_week_training(df, shift_week_num=SHIFT_NUM):
	print('Training data shift:', shift_week_num, 'week(s)')
	df['total_cases'] = df['total_cases'].shift(-1*shift_week_num)
	return df.iloc[:-1*shift_week_num]

def shift_week_testing(df, shift_week_num=SHIFT_NUM):
	print('Testing data shift:', shift_week_num, 'week(s)')
	# if not run_training:
	# 	global sj_train, iq_train
	# 	sj_train, iq_train = preprocess_data(train_feature_path, labels_path=train_label_path)
	df = df.shift(shift_week_num)
	df.fillna(method='bfill', inplace=True)

	return df

def add_monthly_mean():
	return None

def add_rolling_mean():
	return None

def moving_avg(data, n=3):
	ret = np.cumsum(data)
	ret[n::] = ret[n::] - ret[:-n]
	ret = np.hstack((data[:n-1],ret[n-1::]/n))
	return ret

def rolling_cross_validation(data, model, cv_split=5, window_size=3):
	tscv = TimeSeriesSplit(n_splits=cv_split)
	err = []

	for train_idx, test_idx in tscv.split(data):
		train = data.iloc[train_idx]
		test = data.iloc[test_idx]

		model.fit(train[pred], train[target])
		result = model.predict(test[pred])
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
	return rolling_cross_validation(data, model, window_size=WINDOW)

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

	print('\nBest parameters:', best_params, '\n')
	return best_params

def get_best_model(sj_params, iq_params):
	print('Training best models.')
	sj_model = RandomForestRegressor(**sj_params)
	sj_model.fit(sj_train[pred], sj_train[target])
	iq_model = RandomForestRegressor(**iq_params)
	iq_model.fit(iq_train[pred], iq_train[target])

	with open('./rfr_models/sj_rfr_model.pickle','wb') as f:
		pickle.dump(sj_model, f)
	with open('./rfr_models/iq_rfr_model.pickle','wb') as f:
		pickle.dump(iq_model, f)

def training(data_path, labels_path):
	print('Start training...')

	### Read training data
	global sj_train, iq_train
	sj_train, iq_train = preprocess_data(data_path, labels_path=labels_path)

	### Shift feature week
	if SHIFT_WEEK:
		sj_train = shift_week_training(sj_train)
		iq_train = shift_week_training(iq_train)

	### Add lagging data
	if ADD_LAGGING:
		sj_train = add_lagging_data(sj_train)
		iq_train = add_lagging_data(iq_train)

	### Shuffle
	# sj_train = sj_train.sample(frac=1)
	# iq_train = iq_train.sample(frac=1)

	### Set predictor
	set_predictor()
	print(pred)

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

	print('MAE:', (sj_best*936 + iq_best*520) / 1456, '\n')

	### Get best model
	get_best_model(sj_params, iq_params)

	print('Training is done.')

def testing(data_path, sub_path, pred_path, sj_model_path, iq_model_path):
	print('Start testing...')

	sj_test, iq_test = preprocess_data(data_path)

	### Shift feature week
	if SHIFT_WEEK:
		sj_test = shift_week_testing(sj_test)
		iq_test = shift_week_testing(iq_test)

	### Add lagging data
	if ADD_LAGGING:
		sj_test = add_lagging_data(sj_test)
		iq_test = add_lagging_data(iq_test)
		
	set_predictor()

	with open(sj_model_path,'rb') as f:
		sj_model = pickle.load(f)
	print('Loading sj model from: ', sj_model_path)
	with open(iq_model_path,'rb') as f:
		iq_model = pickle.load(f)
	print('Loading iq model from: ', iq_model_path)

	sj_predictions = sj_model.predict(sj_test[pred]).astype(int)
	iq_predictions = iq_model.predict(iq_test[pred]).astype(int)

	# print('Moving average window size:', WINDOW)
	# sj_predictions = np.round(moving_avg(sj_predictions, n=WINDOW)).astype(int)
	# iq_predictions = np.round(moving_avg(iq_predictions, n=WINDOW)).astype(int)
	
	submission = pd.read_csv(sub_path, index_col=[0, 1, 2])

	print('Save predictions to', pred_path)
	submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
	submission.to_csv(pred_path)

	print('Testing is done.')

def main():

	### Training
	if run_training:
		training(train_feature_path, train_label_path)

	### Testing
	if run_testing:
		sj_model_path = './rfr_models/18-sj_rfr.pickle'
		iq_model_path = './rfr_models/18-iq_rfr.pickle'
		testing(test_feature_path, submission_path, prediction_path, sj_model_path, iq_model_path)

if __name__=='__main__':
	start_time = time.time()
	main()
	print('Elapse time:', time.time()-start_time, 'seconds\n')
