import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def method_eval(pred_df, names, mos):
    plcc_all, srcc_all, krcc_all, rmse_all = [], [], [], []
    method_mos = {}
    for name in names:
        subid = name.split('-')[-2]
        if subid not in method_mos.keys():
            method_mos[subid] = mos[names.index(name)]
    
    indexs = list(pred_df['index'])
    for k in range(len(indexs)):
        index_ = indexs[k].split('\'')
        index_ = [index_[1], index_[3], index_[5]]
        pred = {}
        for i in index_: pred[i] = []
        for name in names:
            subid = name.split('-')[-2]
            if subid in pred.keys():
                pred[subid].append(list(pred_df[name])[k])

        pred_mos = []
        gt_mos = []

        for subid in pred.keys():
            pred_mos.append(np.array(pred[subid]).mean())
            gt_mos.append(method_mos[subid])
        
        pred_mos = np.array(pred_mos)
        gt_mos = np.array(gt_mos)

        srcc = scipy.stats.spearmanr(pred_mos, gt_mos)[0]
        krcc = scipy.stats.kendalltau(pred_mos, gt_mos)[0]
        plcc = scipy.stats.pearsonr(pred_mos, gt_mos)[0]
        rmse = np.sqrt(mean_squared_error(pred_mos, gt_mos))
        plcc_all.append(plcc)
        rmse_all.append(rmse)
        srcc_all.append(srcc)
        krcc_all.append(krcc)
        print(k+1, 'th iter, plcc = ', plcc)
    
    return plcc_all, rmse_all, srcc_all, krcc_all

def eval(pred_df, names, mos):
    plcc_all, srcc_all, krcc_all, rmse_all = [], [], [], []
    for k in range(len(list(pred_df['index']))):
        gt_mos = []
        pred_mos = []
        for name in names:
            if not np.isnan(pred_df[name][k]):
                pred_mos.append(pred_df[name][k])
                gt_mos.append(mos[names.index(name)])
        
        srcc_all.append(scipy.stats.spearmanr(pred_mos, gt_mos)[0])
        krcc_all.append(scipy.stats.kendalltau(pred_mos, gt_mos)[0])
        plcc_all.append(scipy.stats.pearsonr(pred_mos, gt_mos)[0])
        rmse_all.append(np.sqrt(mean_squared_error(pred_mos, gt_mos)))
        

    return plcc_all, srcc_all, krcc_all, rmse_all
        


if __name__=='__main__':

    mos_file = r'./metadata/c3_crop_metadata_withmos.csv'   # video level
    # mos_file = r'./metadata/c3_crop_metadata_withmos_subid.csv' for method level evaluation
    pred_file = r'./predict/DFGC-C3_ResNet50_withstd_feats160_pred_log_subid.csv'

    pred_df = pd.read_csv(pred_file)

    mos_df = pd.read_csv(mos_file)
    names = list(mos_df['name'])
    mos = np.array(list(mos_df['mos']), dtype=np.float)

    # evaluate the predictions. use method_eval with method ground truth to get method-level evaluation
    plcc_all, srcc_all, krcc_all, rmse_all = eval(pred_df, names, mos)

    srcc_ave = np.array(srcc_all).mean()
    krcc_ave = np.array(krcc_all).mean()
    plcc_ave = np.array(plcc_all).mean()
    rmse_ave = np.array(rmse_all).mean()
    srcc_std = np.array(srcc_all).std()
    krcc_std = np.array(krcc_all).std()
    plcc_std = np.array(plcc_all).std()
    rmse_std = np.array(rmse_all).std()

    print('\n\n')
    print('======================================================')
    print('Average testing results among all repeated 80-20 holdouts:')
    print('SRCC: ',srcc_ave,'( std:',srcc_std,')')
    print('KRCC: ',krcc_ave,'( std:',krcc_std,')')
    print('PLCC: ',plcc_ave,'( std:',plcc_std,')')
    print('RMSE: ',rmse_ave,'( std:',rmse_std,')')
    print('======================================================')
    print('\n\n')
