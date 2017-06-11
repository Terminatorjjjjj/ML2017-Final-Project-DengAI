#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
file1 = "12.981128_2.835006_1_100_0.05_0.4_1.5.csv"
file2 = "12.868151_3.756847_2_50_0.4_0.6_2.0.csv"
file3 = "12.962234_6.349027_1_100_0.01_0.4_1.5.csv"
file4 = "13.129943_5.357238_2_75_0.2_0.6_2.0.csv"
test_feature_file = 'test_feature.csv'
result_path = 'pred.csv'

test_feature = pd.read_csv(test_feature_file, encoding='big5')
test_feature = test_feature.values
test_feature = test_feature.astype(str)
test_tags = np.asarray(test_feature[:,:3])


load1 = pd.read_csv(file1, encoding='big5')
load2 = pd.read_csv(file2, encoding='big5')
load3 = pd.read_csv(file3, encoding='big5')
load4 = pd.read_csv(file4, encoding='big5')

pred = np.around((((load1.values[:,3] + load2.values[:,3] + load3.values[:,3] + load4.values[:,3])/4).astype(float)))


output = open(result_path, 'w')
output.write('city,year,weekofyear,total_cases\n')
for i in range(len(pred)):
    line = ''
    for j in range(3):
        line += str(test_tags[i,j]) + ','
    line += str(int(pred[i])) + '\n'
    output.write(line)
output.close()
