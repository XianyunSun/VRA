import numpy as np
import pandas as pd
import scipy.io
import os
from selector import *

# ================== select features with given dimention ==========================


if __name__=='__main__':

    feats_path = r'./features'
    label_path = r'./metadata/c3_crop_metadata_withmos.csv'
    out_path = r'./features_selected/'
    phase = 'C3' # C1, C2 or C3

    os.makedirs(out_path, exist_ok=True)

    algo_name = 'ResNet50_withstd'
    num_workers = 5
    num_iters = 100
    num_feats = 280  # dimension of the selected features
    
    # load features and ground truth mos
    feats_file = os.path.join(feats_path, 'DFGC-'+phase+'_'+algo_name+'_feats.mat')
    feats = scipy.io.loadmat(feats_file)
    feats = np.asarray(feats['feats_mat'], dtype=np.float)
    mos_df = pd.read_csv(label_path)
    mos_gt = np.array(list(mos_df['mos']), dtype=np.float)

    print('loaded feature mat shape:', feats.shape)
    
    # select features and save selected index
    _, mask = feat_sel_eval(feats, mos_gt, out_path, phase, algo_name, num_feats, num_iters, num_workers)

    # save selected feature mat
    pick_out(feats, mask, algo_name, phase, num_feats, out_path)

