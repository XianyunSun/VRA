from sys import path
import warnings
import time
import os
import pandas as pd
import numpy as np
import scipy.stats
import scipy.io
from sklearn import model_selection
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
# ignore all warnings
warnings.filterwarnings("ignore")


##########################################################################
###### read multiple feature files and return X

def read_feats(feat_path, algo_name, phase='C3'):
  X = []
  
  for name in algo_name:
    mat_file = os.path.join(feat_path, str('DFGC-'+phase+'_'+name+'_feats.mat'))
    X_mat = scipy.io.loadmat(mat_file)
    X_mat = np.asarray(X_mat['feats_mat'], dtype=np.float)
    X.append(X_mat)
    print('name:', name, ', shape:', X_mat.shape)
  
  X = np.hstack(X)
  X[np.isnan(X)] = 0
  X[np.isinf(X)] = 0
  
  return X

##########################################################################
###### pick out the corresponding features after a selection, output: .mat file

def pick_out(X, feats, algo_name, phase, num_feat, out_path):
  out_file = os.path.join(out_path, str('DFGC-'+phase+'_'+algo_name+'_feats'+str(num_feat)+'.mat'))

  X_reduced = X[:,feats]
  feats = np.asarray(feats, dtype=np.int)

  dim = X.shape[1]
  print('num of feats in mean:', np.count_nonzero(feats<dim/2))
  print('num of feats in std:', np.count_nonzero(feats<dim)-np.count_nonzero(feats<dim/2))

  scipy.io.savemat(out_file, {'feats_mat':X_reduced})


##########################################################################
###### evaluation with given train & test sets 

def compute_metrics(y_pred, y):
  # compute SRCC & KRCC
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  PLCC = scipy.stats.pearsonr(y, y_pred)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred))
  return [SRCC, KRCC, PLCC, RMSE]

### eval features with svr
def eval_with_mask(X, y, mask_vec, C=pow(2,6), gamma=0.1, test_size=0.2):
  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
  X_train_reduced = X_train[:,mask_vec]
  X_test_reduced = X_test[:,mask_vec]

  model = SVR(kernel='rbf', gamma=gamma, C=C)
  scaler = preprocessing.MinMaxScaler().fit(X_train_reduced)
  X_train_reduced = scaler.transform(X_train_reduced)
  X_test_reduced = scaler.transform(X_test_reduced) 
  
  model.fit(X_train_reduced, y_train)
  
  y_train_pred = model.predict(X_train_reduced)
  y_test_pred = model.predict(X_test_reduced)
  metrics_train = compute_metrics(y_train_pred, y_train)
  metrics_test = compute_metrics(y_test_pred, y_test)

  result = metrics_train + metrics_test

  return result


##########################################################################
###### selector function: select features given feature group and corresponding mos with fixed parameters
def feature_selection_single_svr(X, y, num_feats, C, gamma, test_size=0.2, seed=42):

  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
    
  model = SVR(kernel='linear', C=C, gamma=gamma)
    
  scaler_sel = preprocessing.MinMaxScaler().fit(X_train)
  X_train_sel = scaler_sel.transform(X_train)  
    
  model.fit(X_train_sel, y_train)
  importances = np.abs(model.coef_[0])

  indices = np.argsort(importances)[::-1]
  mask = indices[0:num_feats]

  return mask  

##########################################################################
###### select feature number
def param_selection_fnum_svr(X, y, algo_name, phase, out_path, num_feats, num_iters, num_workers, param = [pow(2,6), 0.1]):
  
  out_path = os.path.join(out_path, str('DFGC-'+phase+'_'+algo_name+'_numsel_eval.csv'))
  result_all = []
  head = ['num_feat', 'SRCC_train', 'KRCC_train', 'PLCC_train', 'RMSE_train', 'SRCC_test', 'KRCC_test', 'PLCC_test', 'RMSE_test']
  best_plcc = -1
  best_num = 0

  for num_feat in num_feats:
    result = Parallel(n_jobs=num_workers)(delayed(feature_sel_eval_single)(i, X, y, num_feat) for i in range(num_iters))
    
    eval_result = []
    for result_single in result:
      eval_result.append(np.asarray(result_single['result'], dtype=np.float))

    eval_result = np.nanmean(np.asarray(eval_result), axis=0)
    eval_result = [num_feat]+list(eval_result)
    result_all.append(eval_result)
    print('feats num:', num_feat, ', PLCC test:', eval_result[7])
    if eval_result[7]>best_plcc:
      best_plcc = eval_result[7]
      best_num = num_feat  

  result_df = pd.DataFrame(result_all, columns=head)
  result_df.to_csv(out_path, index=False)

  print('best test PLCC:', best_plcc, ', with feats num:', best_num)

    
##########################################################################
###### feature selection with evaluation under fixed parameters

def feature_sel_eval_single(i, X, y, num_feat, param=[pow(2,6), 0.1]):   # <<< param: for svr feature selector(if used), param=[C, gamma]

  t0 = time.time()
  out = {}

  indices = feature_selection_single_svr(X, y, num_feat,C=param[0], gamma=param[1], seed=i)

  mask_vec = np.zeros(X.shape[1])
  mask_vec[indices] = 1
  mask_vec = np.array(mask_vec, dtype='int16')

  result = eval_with_mask(X, y, indices)    #  <<< evaluation with svr using defult parameters, set parameter values here if needed

  out['result'] = result
  out['mask'] = mask_vec

  print("{} th iterations finished with {} features! {} secs elapsed...".format(i+1, num_feat, str(time.time() - t0)))

  return out


def feat_sel_eval(X, y, out_path, phase, algo_name, num_feat, num_iters, num_workers, param=[pow(2,6), 0.1]):

  out_mask_path = os.path.join(out_path, str('DFGC-'+phase+'_'+algo_name+'_feats'+str(num_feat)+'_sel-mask.csv'))

  masks = []

  out = Parallel(n_jobs=num_workers)(delayed(feature_sel_eval_single)(i, X, y, num_feat) for i in range(num_iters))

  for out_single in out:
    masks.append(out_single['mask'].tolist())

  masks_df = pd.DataFrame(zip(*list(masks)))

  masks_df.to_csv(out_mask_path, index=False, header=False)

  masks = np.asarray(masks, dtype=int)
  masks_sum = np.sum(masks, axis=0)
  masks_index = np.argsort(masks_sum)[-num_feat:]
  masks_index = masks_index.tolist()
  masks_index.sort()

  index_save = os.path.join(out_path, str(algo_name+'_feats'+str(num_feat)+'_sel-mask.npy'))
  np.save(index_save, np.asarray(masks_index, dtype=np.int))

  return masks_sum, masks_index



    
