#!/usr/bin/env python3
#coding=utf-8

import time
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import pickle

train_feature_path = sys.argv[1]
train_label_path = sys.argv[2]
test_feature_path = sys.argv[3]
submission_path = sys.argv[4]
prediction_path = sys.argv[5]

################
###   Util   ###
################
def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']
    df = df[features]
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add square terms
    new_features = ['reanalysis_specific_humidity_g_per_kg_2', 
                 'reanalysis_dew_point_temp_k_2', 
                 'station_avg_temp_c_2', 
                 'station_min_temp_c_2']
    for f in range(4):
        df[new_features[f]] = df[features[f]] ** 2

    # # add cube terms
    # new_features = ['reanalysis_specific_humidity_g_per_kg_3', 
    #              'reanalysis_dew_point_temp_k_3', 
    #              'station_avg_temp_c_3', 
    #              'station_min_temp_c_3']
    # for f in range(4):
    #     df[new_features[f]] = df[features[f]] ** 3

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    tmp = sj.iloc[:,0:8]
    print(sj.shape)
    print(tmp.shape)
    
    return sj, iq

def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c + " \
                    "reanalysis_specific_humidity_g_per_kg_2 + " \
                    "reanalysis_dew_point_temp_k_2 + " \
                    "station_min_temp_c_2 + " \
                    "station_avg_temp_c_2"
                    # "reanalysis_specific_humidity_g_per_kg_3 + " \
                    # "reanalysis_dew_point_temp_k_3 + " \
                    # "station_min_temp_c_3 + " \
                    # "station_avg_temp_c_3"
    
    grid = np.append(10 ** np.arange(-8, -3, dtype=np.float64), \
    				5 * (8 ** np.arange(-8, -3, dtype=np.float64)))
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return (fitted_model, best_alpha, best_score)

#########################
###   Main function   ###
#########################
def main():

    ### Read training data
    sj_train, iq_train = preprocess_data(train_feature_path, labels_path=train_label_path)

    ### Split validation
    # sj_train = sj_train.sample(frac=1, replace=True)
    sj_train_subtrain = sj_train.head(800) # 842
    sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

    # iq_train = iq_train.sample(frac=1, replace=True)
    iq_train_subtrain = iq_train.head(400) # 468
    iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

    ### Training
    (sj_best_model, sj_best_alpha, sj_best_score) = get_best_model(sj_train_subtrain, sj_train_subtest)
    (iq_best_model, iq_best_alpha, iq_best_score) = get_best_model(iq_train_subtrain, iq_train_subtest)
    with open('sj_best_model.pickle', 'wb') as sjm:
        pickle.dump(sj_best_model, sjm)
    with open('iq_best_model.pickle', 'wb') as iqm:
        pickle.dump(iq_best_model, iqm)

    print('average mae = ', (936*sj_best_score + 520*iq_best_score) / 1456)

    ### Testing
    sj_test, iq_test = preprocess_data(test_feature_path)

    sj_predictions = sj_best_model.predict(sj_test).astype(int)
    iq_predictions = iq_best_model.predict(iq_test).astype(int)

    submission = pd.read_csv(submission_path, index_col=[0, 1, 2])

    submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
    submission.to_csv(prediction_path)

if __name__=='__main__':
    start_time = time.time()
    main()
    print('Elapse time:', time.time()-start_time, 'seconds\n')

