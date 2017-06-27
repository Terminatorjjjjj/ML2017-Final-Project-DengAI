#!/usr/bin/env python3
#coding=utf-8

import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

train_feature_path = './data/train_feature.csv'
test_feature_path = './data/test_feature.csv'
prediction_path = './pred_merge.csv'
merge_path = './pred_files/rnn2221.csv'
merge_path_2 = './pred_files/arc.csv'

WINDOW = 3
MERGE_WEEKS = 20

full_pred = ['week',
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
veg_pred = ['ndvi_ne',
			'ndvi_nw',
			'ndvi_se',
			'ndvi_sw']
core_pred = ['reanalysis_specific_humidity_g_per_kg', 
			'reanalysis_dew_point_temp_k', 
			'station_avg_temp_c', 
			'station_min_temp_c',
			'week']
target = 'total_cases' 

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

def merge_feature(sj_train, iq_train, sj_test, iq_test):
	sj_test_ret = pd.concat(objs=[sj_train[-1*MERGE_WEEKS::], sj_test])
	iq_test_ret = pd.concat(objs=[iq_train[-1*MERGE_WEEKS::], iq_test])
	return sj_test_ret, iq_test_ret

def add_lagging(df, lagging_num, pred, lagging_pred):
	print('Lagging', lagging_num, 'week(s)')
	ret_df = df.copy()
	ret_pred = list(pred)
	features = list(lagging_pred)

	# add lagging data and predictor names
	for feature in features:
		for i in range(lagging_num):
			lagging_feature = feature + '_lag' + str(i+1)
			ret_df[lagging_feature] = df[feature].shift(i+1)
			if lagging_feature not in ret_pred:
				ret_pred.append(lagging_feature)

	ret_df.fillna(method='bfill', inplace=True)

	return ret_df, ret_pred

def add_shift(df, shift_week_num):
	print('Shift', shift_week_num, 'week(s)')
	ret_df = df.copy()
	ret_df = ret_df.shift(shift_week_num)
	ret_df.fillna(method='bfill', inplace=True)

	return ret_df

def moving_avg(data, n=3):
	ret = np.cumsum(data)
	ret[n::] = ret[n::] - ret[:-n]
	ret = np.hstack((data[:n-1],ret[n-1::]/n))
	return ret

def make_prediction(data, pred, model_path, shift_week=0, lagging_week=0):
	test = data.copy()
	predictor = pred.copy()

	# shift features
	if shift_week:
		test = add_shift(test, shift_week)
	# add lagging features
	if lagging_week:
		test, predictor = add_lagging(test, lagging_week, pred, pred)
	# remove training data part
	test = test[MERGE_WEEKS::]	

	print('Loading model from: ', model_path)
	with open(model_path,'rb') as f:
		model = pickle.load(f)

	result = model.predict(test[predictor]).astype(float)

	return result

def make_submission(result, id_path=test_feature_path, pred_path=prediction_path):
	submission = pd.read_csv(id_path, index_col=[0, 1, 2])
	prediction = pd.DataFrame(index=submission.index.copy())
	prediction[target] = result
	prediction.to_csv(pred_path)
	print('Save predictions to', pred_path)

def plot_results(result, rnn_result, arc_result):
	print('Plot results.')
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.plot(result, color='b')
	plt.plot(rnn_result, color='r')
	plt.plot(arc_result, color='g')
	plt.title('Prediction Results')
	plt.xlabel('Weeks')
	plt.ylabel('Total Cases')
	plt.legend(['RFR w/o lagging labels', 'RNN', 'RFR w/ lagging labels'], loc='upper right')
	fig.savefig('results.png')

def plot_result(result):
	print('Plot ensemble result.')
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.plot(result, color='b')
	plt.title('Ensemble Result')
	plt.xlabel('Weeks')
	plt.ylabel('Total Cases')
	fig.savefig('ensemble.png')

def main():
	print('Start testing...')

	### Read training and testing data
	sj_train, iq_train = preprocess_data(train_feature_path)
	sj_test, iq_test = preprocess_data(test_feature_path)

	### Merge last MERGE_WEEKS training to testing features for shifting and lagging
	sj_test, iq_test = merge_feature(sj_train, iq_train, sj_test, iq_test)

	### Testing sj
	print('Testing sj...')
	params_1 = {'data': sj_test,
				'pred': core_pred,
				'model_path': '../rfr_models/18-sj_rfr.pickle',
				'shift_week': 0,
				'lagging_week': 9}
	result_1 = make_prediction(**params_1)

	sj_result = result_1

	### Testing iq
	print('Testing iq...')
	params_1 = {'data': iq_test,
				'pred': core_pred,
				'model_path': '../rfr_models/18-iq_rfr.pickle',
				'shift_week': 0,
				'lagging_week': 9}
	result_1 = make_prediction(**params_1)

	iq_result = result_1

	### Moving average
	print('Moving average window size:', WINDOW)
	sj_result = np.round(moving_avg(sj_result, n=WINDOW)).astype(int)
	iq_result = np.round(moving_avg(iq_result, n=WINDOW)).astype(int)
	result = np.concatenate([sj_result, iq_result])

	### Merge with rnn
	print('Load merge file:', merge_path)
	rnn = pd.read_csv(merge_path, index_col=[0, 1, 2])
	rnn_result = rnn[target].values
	rnn_sj = rnn_result[:260]
	rnn_iq = rnn_result[260::]
	print('Load merge file:', merge_path_2)
	arc = pd.read_csv(merge_path_2, index_col=[0, 1, 2])
	arc_result = arc[target].values
	arc_sj = arc_result[:260]
	arc_iq = arc_result[260::]

	### Plot results
	plot_results(result, rnn_result, arc_result.astype(int))

	sj_result = 0.4*sj_result + 0.4*rnn_sj + 0.2*arc_sj
	# sj_result = 0.4*sj_result + 0.6*rnn_sj
	iq_result = 0.6*iq_result + 0.2*rnn_iq + 0.2*arc_iq

	# result = 0.4*result + 0.4*rnn_result + 0.2*arc_result
	result = np.concatenate([sj_result, iq_result])
	plot_result(result)

	### Make submission	
	make_submission(result.astype(int))

	print('Testing is done.')

if __name__=='__main__':
	start_time = time.time()
	main()
	print('Elapse time:', time.time()-start_time, 'seconds\n')