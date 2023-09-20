import numpy as np
import pandas as pd
import scipy.io
import os
from selector import *

# ================== select dimention of features ==========================


if __name__=='__main__':

    feats_path = r'./features/'
    label_path = r'./metadata/c3_crop_metadata_withmos.csv'
    out_path = r'./features_selected_num/'
    out_name = 'ResNet50_withstd'
    phase = 'C3' # C1, C2 or C3

    os.makedirs(out_path, exist_ok=True)

    num_workers = 5
    num_iters = 10
    num_feats = list(range(20, 520, 20)) # grid search for the best dimension of the selected features

    # load features
    # algo_name is a list of featurs you want to fuse and to choose from. 
    algo_name = ['ResNet50_mean', 'ResNet50_std']

    feats = read_feats(feats_path, algo_name, phase)
    print('loaded feature mat shape:', feats.shape)
 
    # load ground truth mos
    mos_df = pd.read_csv(label_path)
    mos_gt = np.array(list(mos_df['mos']), dtype=np.float)

    param_selection_fnum_svr(feats, mos_gt, out_name, phase, out_path, num_feats, num_iters, num_workers)
