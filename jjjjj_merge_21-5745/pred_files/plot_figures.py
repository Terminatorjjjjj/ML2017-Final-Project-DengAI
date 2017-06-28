#!/usr/bin/env python3
#coding=utf-8

import sys
import time
import numpy as np
import pandas as pd
import pickle

rfr_path = './pred_rfr.csv'
rnn_path = './rnn2221.csv'
arc_path = './arc.csv'

def read_data(data_path):
	print('Reading data from:', data_path)
	# load data and set index to city, year, weekofyear
	df = pd.read_csv(data_path, index_col=[0, 1, 2])
	sj = df.loc['sj']
	iq = df.loc['iq']
	return (sj['total_cases'].values, iq['total_cases'].values, df['total_cases'].values)

def plot_results(result, rnn_result, arc_result, title, name):
	print('Plot results.')
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.plot(result, color='b')
	plt.plot(rnn_result, color='r')
	plt.plot(arc_result, color='g')
	plt.title(title)
	plt.xlabel('Weeks')
	plt.ylabel('Total Cases')
	plt.legend(['RFR w/o lagging labels', 'RNN', 'RFR w/ lagging labels'], loc='upper right')
	fig.savefig(name + '.png')

def plot_result(result, title, name):
	print('Plot ensemble result.')
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.plot(result, color='b')
	plt.title(title)
	plt.xlabel('Weeks')
	plt.ylabel('Total Cases')
	fig.savefig(name + '.png')

def main():
	(rfr_sj, rfr_iq, rfr_result) = read_data(rfr_path)
	(rnn_sj, rnn_iq, rnn_result) = read_data(rnn_path)
	(arc_sj, arc_iq, arc_result) = read_data(arc_path)
	(train_sj, train_iq, train) = read_data('../data/train_label.csv')

	### Plot trainging labels
	# plot_result(train, 'Training Labels', 'train_label')
	# plot_result(train_sj, 'Training Labels: sj', 'train_sj')
	# plot_result(train_iq, 'Training Labels: iq', 'train_iq')

	### Plot ensemble result 451 523

	sj_result = rfr_sj
	for i in range(rfr_sj.shape[0]):
		if arc_sj[i] - 0.5*rfr_sj[i] - 0.5*rnn_sj[i] >= 90:
			sj_result[i] = 0.5*rfr_sj[i] + 0.5*rnn_sj[i]
		else:
			sj_result[i] = 0.4*rfr_sj[i] + 0.4*rnn_sj[i] + 0.2*arc_sj[i]
	# sj_result = 0.3*rfr_sj + 0.5*rnn_sj + 0.2*arc_sj
	# iq_result = rfr_iq
	iq_result = 0.4*rfr_iq + 0.4*rnn_iq + 0.2*arc_iq
	result = np.concatenate([sj_result, iq_result])
	plot_result(result, 'Ensembel Result', 'ensemble')

	### Plot predictions comparison
	# plot_results(rfr_sj, rnn_sj, arc_sj, 'Prediction Results: sj', 'results-sj')
	# plot_results(rfr_iq, rnn_iq, arc_iq, 'Prediction Results: iq', 'results-iq')
	# plot_results(rfr_result, rnn_result, arc_result, 'Prediction Results', 'results')

if __name__=='__main__':
	start_time = time.time()
	main()
	print('Elapse time:', time.time()-start_time, 'seconds\n')
