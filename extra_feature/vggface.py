# -*- coding: utf-8 -*-
## this code should be put in the 'model' folder

from logging import raiseExceptions
import cv2
import torch
import numpy as np
from vgg_face import VGG_16
import pandas as pd
import sys
import os
import cv2
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as scio



if __name__ == '__main__':
    model = VGG_16().float()
    model.load_weights()

    device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
    video_path = r'../../../C3-cropped'
    meta_file = r'../../../metadata/c3_crop_metadata_withmos.csv'
    out_path = r'../../../features/VGGface'
    out_name = 'DFGC-C3_VGGface'

    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(meta_file)
    framerate = list(df['framerate'])
    video_list = list(df['name'])

    model.to(device)
    print('VGG-Face model loaded')

    feats_mat_mean = []
    feats_mat_std = []
    with torch.no_grad():
        for i in tqdm(range(len(video_list))):
            camera = cv2.VideoCapture(os.path.join(video_path, video_list[i]) + '.mp4')
            feats_frames = []
            times = 0
            while True:
                times += 1
                res, frame = camera.read()
                if not res:
                    break
                if times % (framerate[i]//6) == 0:
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    frame_Image = Image.fromarray(frame)
                    frame_tensor = transforms.ToTensor()(frame_Image)
                    frame_tensor = transforms.Resize(224)(frame_tensor)
                    frame_tensor = Variable(torch.unsqueeze(frame_tensor, dim=0), requires_grad=False)
                    frame_tensor = frame_tensor.to(device)
                    _, x = model(frame_tensor)
                    x = x.squeeze()
                    feats_frames.append(x)
            feats_frames = torch.stack(feats_frames,dim=0)
            #out_file = os.path.join(out_path, str(video_list[i]+'.npy'))
            #np.save(out_file, feats_frames.cpu().numpy())
            feats_frames_mean = feats_frames.mean(0)
            feats_frames_std = feats_frames.std(0)
            #feats_frames_diff = torch.abs(feats_frames[1:,:]-feats_frames[:-1,:]).mean(0)
            #feats_frames = torch.cat((feats_frames_mean, feats_frames_std), dim=-1)
            #feats_mat.append(feats_frames_diff.cpu().numpy())
            feats_mat_mean.append(feats_frames_mean.cpu().numpy())
            feats_mat_std.append(feats_frames_std.cpu().numpy())

    feats_mat_mean = np.vstack(feats_mat_mean)
    feats_mat_std = np.vstack(feats_mat_std)
    
    out_mean = os.path.join(out_path, out_name+'_mean_feats.mat')
    out_std = os.path.join(out_path, out_name+'DFGC-C2_target_VGGFace_std_feats.mat')
    scio.savemat(out_mean, {'feats_mat':feats_mat_mean})
    scio.savemat(out_std, {'feats_mat':feats_mat_std})