import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
from sklearn import model_selection
import os
import warnings
import time
import pandas
import math
import random as rnd
import scipy.stats
import scipy.io
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# ignore all warnings
warnings.filterwarnings("ignore")


'''======================== parameters ================================''' 
'''
cross_flag=True: inter-subset tests. Train on C3 and test on C1 or C2. 
                 The whole C3 will be used for training, please set split_type=all
cross_flag=False: intra-subset tests. Both train and test on C3.
    split_type='subid': the videos used for train and test will be subid-disjointed, while each subid repersent a different face-swap method
    split_type='id20': the videos used for train and test will be id-disjointed

the 'pred' output is the raw prediction value of the model.
the 'pred_log' output is the prediction after performing a 5-parameter fitting for VQA methods
'''

split_type = 'subid'
cross_flag = False
rnd.seed(42)

if split_type=='all' or (split_type=='subid' and cross_flag):
    iters = 1
elif split_type=='id20' or (split_type=='subid' and not cross_flag):
    iters = 100

train_csv_file = r'./metadata/c3_crop_metadata_withmos.csv'
train_mat_file = r'./features_selected/DFGC-C3_ResNet50_withstd_feats160.mat'

test_csv_file = r'./c3_crop_metadata_withmos.csv'
test_mat_file = r'./features_selected/DFGC-C3_ResNet50_withstd_feats160.mat'

result_file = r'./predict/DFGC-C3_ResNet50_withstd_feats160_pred_subid.csv'
result_log_file = r'./predict/DFGC-C3_ResNet50_withstd_feats160_pred_log_subid.csv'


'''======================== read files =============================== '''
# read training data in c3
df_train = pandas.read_csv(train_csv_file, skiprows=[])
names_train = list(df_train['name'])

y_train_raw = np.array(list(df_train['mos']), dtype=np.float)

X_mat_train_raw = scipy.io.loadmat(train_mat_file)
X_train_raw = np.asarray(X_mat_train_raw['feats_mat'], dtype=np.float)
X_train_raw[np.isnan(X_train_raw)] = 0
X_train_raw[np.isinf(X_train_raw)] = 0

# read test data in c1/c2 (if used)
df_test = pandas.read_csv(test_csv_file, skiprows=[])
names_test = list(df_test['name'])

y_test_raw = np.array(list(df_test['mos']), dtype=np.float)

X_mat_test_raw = scipy.io.loadmat(test_mat_file)
X_test_raw = np.asarray(X_mat_test_raw['feats_mat'], dtype=np.float)
X_test_raw[np.isnan(X_test_raw)] = 0
X_test_raw[np.isinf(X_test_raw)] = 0


'''======================== Split dataset ===========================''' 
### 16 subids for c3
subids3 = ['00000', '91740', '92068', '92069', '92147', '92582', '92584', '93014', '93056', '93059', '93060', '93062', '93065', '93110', '93169', '93170'] 

id20s = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '13', '14', '15', '16', '18', '19', '20', '21', '22']

# choose index for test set
indexs = []
for _ in range(iters):
    if split_type == 'id20':
        index = rnd.sample(id20s, 4)
        indexs.append(index)
    elif split_type == 'subid':
        index = rnd.sample(subids3, 3)
        indexs.append(index)


'''======================== Main Body ===========================''' 

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.
# 

C_range = np.logspace(1, 10, 10, base=2)
gamma_range = np.logspace(-8, 1, 10, base=2)
params_grid = dict(gamma=gamma_range, C=C_range)

y_test_pred_all = {} # <<< dict:{name:[predict score for each iter]}
y_test_pred_log_all = {}
if not split_type=='all':
    y_test_pred_all['index'] = []
    y_test_pred_log_all['index'] = []

for name in names_test:
    y_test_pred_all[name] = []
    y_test_pred_log_all[name] = []


'''======================== Begin iter ===========================''' 
# 100 random splits
for i in range(1, iters+1):
    t0 = time.time()
    # parameters for each hold out
    model_params_all = []
    RMSE_all_test = []
    name_test_iter = []
    
    '''======================== split data ===========================''' 
    
    X_test = []
    y_test = []
    X_train = []
    y_train = []

    if split_type=='id20':
        # add train data
        for j in range(len(names_train)):
            if split_type == 'id20': name = names_train[j].split('-')[0]
            if str(name) not in indexs[i-1]:
                X_train.append(X_train_raw[j])
                y_train.append(y_train_raw[j])
        for k in range(len(names_test)):
            if split_type == 'id20': name = names_test[k].split('-')[0] 
            if str(name) in indexs[i-1]:
                X_test.append(X_test_raw[k])
                y_test.append(y_test_raw[k])
                name_test_iter.append(names_test[k])
        y_test_pred_all['index'].append(indexs[i-1])
        y_test_pred_log_all['index'].append(indexs[i-1])
        print('chosen id test set index:', indexs[i-1])
    
    elif split_type=='subid':
        if cross_flag:
            X_train = X_train_raw
            y_train = y_train_raw
            for j in range(len(names_test)):
                name = names_test[j].split('-')[-2]
                if not name=='00000':
                    X_test.append(X_test_raw[j])
                    y_test.append(y_test_raw[j])
                    name_test_iter.append(names_test[j])
        else:
            for j in range(len(names_train)):
                name = names_train[j].split('-')[-2]
                if name not in indexs[i-1]:
                    X_train.append(X_train_raw[j])
                    y_train.append(y_train_raw[j])
                else:
                    X_test.append(X_test_raw[j])
                    y_test.append(y_test_raw[j])
                    name_test_iter.append(names_test[j])
            y_test_pred_all['index'].append(indexs[i-1])
            y_test_pred_log_all['index'].append(indexs[i-1])
            print('chosen subid test set index:', indexs[i-1])               

                
    elif split_type=='all':
        X_train = X_train_raw
        y_train = y_train_raw
        X_test = X_test_raw
        y_test = y_test_raw
        name_test_iter = names_test
    
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    '''======================== training ===========================''' 
    #### SVR grid search in the TRAINING SET ONLY 
    validation_size = 0.2
    X_param_train, X_param_valid, y_param_train, y_param_valid = \
        model_selection.train_test_split(X_train, y_train, test_size=validation_size, random_state=math.ceil(6.6*i))
    # grid search
    for C in C_range:
        for gamma in gamma_range:
            model_params_all.append((C, gamma))
            model = SVR(kernel='rbf', gamma=gamma, C=C)
            # Standard min-max normalization of features
            scaler = MinMaxScaler().fit(X_param_train)
            X_param_train = scaler.transform(X_param_train) 

            # Fit training set to the regression model
            model.fit(X_param_train, y_param_train)

            # Apply scaling                           
            X_param_valid = scaler.transform(X_param_valid)

            # Predict MOS for the validation set
            y_param_valid_pred = model.predict(X_param_valid)
            y_param_train_pred = model.predict(X_param_train)

            # define 4-parameter logistic regression
            def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
                logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
                yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
                return yhat

            y_param_valid = np.array(list(y_param_valid), dtype=np.float)
            try:
                # logistic regression
                beta = [np.max(y_param_valid), np.min(y_param_valid), np.mean(y_param_valid_pred), 0.5]
                popt, _ = curve_fit(logistic_func, y_param_valid_pred, y_param_valid, p0=beta, maxfev=100000000)
                y_param_valid_pred_logistic = logistic_func(y_param_valid_pred, *popt)
            except:
                raise Exception('Fitting logistic function time-out!!')
            rmse_valid_tmp = np.sqrt(mean_squared_error(y_param_valid, y_param_valid_pred_logistic))

            RMSE_all_test.append(rmse_valid_tmp)



    # using the best chosen parameters to test on testing set
    param_idx = np.argmin(np.asarray(RMSE_all_test, dtype=np.float))
    C_opt, gamma_opt = model_params_all[param_idx]
    model = SVR(kernel='rbf', gamma=gamma_opt, C=C_opt)

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)  

    model.fit(X_train, y_train)
    X_test = scaler.transform(X_test)

    # Predict MOS for the test set
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    try:
        # logistic regression
        beta = [np.max(y_test_pred), np.min(y_test_pred), np.mean(y_test_pred), 0.5]
        popt, _ = curve_fit(logistic_func, y_test_pred, y_test, p0=beta, maxfev=100000000)
        y_test_pred_logistic = logistic_func(y_test_pred, *popt)
        #print(popt)
        #np.save(popt_file, popt)
    except:
        raise Exception('Fitting logistic function time-out!!')

    # calculate acc
    plcc_test = scipy.stats.pearsonr(y_test, y_test_pred_logistic)[0]
    print(i, 'th plcc=', plcc_test)

    # record output
    for j in range(len(names_test)):
        if names_test[j] in name_test_iter:
            y_test_pred_all[names_test[j]].append(y_test_pred[name_test_iter.index(names_test[j])])
            y_test_pred_log_all[names_test[j]].append(y_test_pred_logistic[name_test_iter.index(names_test[j])])
        else:
            y_test_pred_all[names_test[j]].append(float('nan'))
            y_test_pred_log_all[names_test[j]].append(float('nan'))
            
    
    print(i,'th iter finished in', time.time()-t0, 'seconds.')
    

'''======================== Save output ===========================''' 
y_test_pred_all_df = pd.DataFrame(y_test_pred_all)
y_test_pred_all_df.to_csv(result_file, index=True)
y_test_pred_log_all_df = pd.DataFrame(y_test_pred_log_all)
y_test_pred_log_all_df.to_csv(result_log_file, index=True)


    


