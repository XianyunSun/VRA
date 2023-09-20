import vbliinds_demo as vbi
import scipy.io as scio
import os
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def vbliinds_feats_single(i, path, name):
    video = os.path.join(path, str(name)+'.mp4')
    t0 = time.time()
    feat = vbi.process_main(video)
    feat = feat.tolist()
    print('finished:', i, ', time:',time.time()-t0, ', path:', video)

    return feat



if __name__=='__main__':
    video_path = r'../../data/C3-cropped'
    meta_file = r'../../metadata/c3_crop_metadata_withmos.csv'
    out_path = r'../../features/vbliinds'
    out_file = os.path.join(out_path, 'DFGC-C3_VBLIINDS_feats.mat')
    num_workers = 6

    names = pd.read_csv(meta_file)
    names = list(names.name)
    num_videos = len(names)

    vbliinds_feats = Parallel(n_jobs=num_workers)(delayed(vbliinds_feats_single)(i, video_path, names[i]) for i in range(num_videos))

    scio.savemat(out_file, mdict={'feats_mat': np.asarray(vbliinds_feats,dtype=np.float)})


