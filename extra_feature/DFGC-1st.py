# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
import cv2
from PIL import Image

from transforms import build_transforms
from data_utils import images_Dataloader, face_Dataloader
from network.models import get_swin_transformers, get_convnext
#from extract_face.extract_video_mtcnn import Extractor

import pandas as pd
import os
import time
import scipy.io as scio
from tqdm import tqdm


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    print(f'Load model in {save_filename}')
    return network


def get_models(model_names, model_paths, device):
    torch.cuda.empty_cache()
    this_dir = os.path.dirname(os.path.abspath(__file__))  # use this line to find this file's dir
    models = []
    for i in range(len(model_names)):
        model_name = model_names[i]
        model_path = model_paths[i]
        if 'swin' in model_name:
            model = get_swin_transformers(model_name=model_name, num_classes=2, pretrained=False)
        else:
            model = get_convnext(model_name=model_name, num_classes=2, pretrained=False)
            model = torch.nn.DataParallel(model)  # 多gpu训练的原因
        model = load_network(model, os.path.join(this_dir, model_path)).to(device)
        model.train(False)
        model.eval()
        models.append(model)

    return models


def ensemble(models, inputs):

    with torch.no_grad():

        out_cv10 = models[0].module.forward_features(inputs)
        out_cv10 = models[0].module.head.global_pool(out_cv10)
        out_cv10 = models[0].module.head.flatten(out_cv10).cpu().detach().numpy()

        out_cv30 = models[1].module.forward_features(inputs)
        out_cv30 = models[1].module.head.global_pool(out_cv30)
        out_cv30 = models[1].module.head.flatten(out_cv30).cpu().detach().numpy()
        
        out_swin = models[2].forward_features(inputs)
        out_swin = models[2].forward_head(out_swin, pre_logits=True).cpu().detach().numpy()

    return out_cv10, out_cv30, out_swin


def feats_all_videos(video_path, meta_file, device, out_path):
    df = pd.read_csv(meta_file)
    video_list = list(df['name'])
    framerate = list(df['framerate'])

    feats_cv10_mean = []
    feats_cv30_mean = []
    feats_swin_mean = []
    feats_cv10_std = []
    feats_cv30_std = []
    feats_swin_std = []
    
    for i in tqdm(range(len(video_list))):
        torch.cuda.empty_cache()

        camera = cv2.VideoCapture(os.path.join(video_path, video_list[i]) + '.mp4')
        feats_frames_cv10 = []
        feats_frames_cv30 = []
        feats_frames_swin = []
        times = 0
        while True:
            times += 1
            res, frame = camera.read()
            if not res: break
            if times % (framerate[i]//6) == 0:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame_Image = Image.fromarray(frame)
                frame_tensor = transforms.ToTensor()(frame_Image)
                frame_tensor = transforms.Resize([384,384])(frame_tensor)
                frame_tensor = Variable(torch.unsqueeze(frame_tensor, dim=0), requires_grad=False)
                frame_tensor = frame_tensor.to(device)
                cv10, cv30, swin = ensemble(models, frame_tensor)
                feats_frames_cv10.append(cv10)
                feats_frames_cv30.append(cv30)
                feats_frames_swin.append(swin)
        feats_frames_cv10_mean = np.vstack(feats_frames_cv10).mean(0)
        feats_frames_cv30_mean = np.vstack(feats_frames_cv30).mean(0)
        feats_frames_swin_mean = np.vstack(feats_frames_swin).mean(0)
        #feats_frames_cv10_diff = np.abs(np.vstack(feats_frames_cv10)[1:,:]-np.vstack(feats_frames_cv10)[:-1,:]).mean(0)
        #feats_frames_cv30_diff = np.abs(np.vstack(feats_frames_cv30)[1:,:]-np.vstack(feats_frames_cv30)[:-1,:]).mean(0)
        #feats_frames_swin_diff = np.abs(np.vstack(feats_frames_swin)[1:,:]-np.vstack(feats_frames_swin)[:-1,:]).mean(0)
        feats_frames_cv10_std = np.vstack(feats_frames_cv10).std(axis=0, ddof=1)
        feats_frames_cv30_std = np.vstack(feats_frames_cv30).std(axis=0, ddof=1)
        feats_frames_swin_std = np.vstack(feats_frames_swin).std(axis=0, ddof=1)
        

        #feats_cv10.append(np.concatenate([feats_frames_cv10_mean, feats_frames_cv10_diff], axis=0))
        #feats_cv30.append(np.concatenate([feats_frames_cv30_mean, feats_frames_cv30_diff], axis=0))
        #feats_swin.append(np.concatenate([feats_frames_swin_mean, feats_frames_swin_diff], axis=0))
        
        feats_cv10_mean.append(feats_frames_cv10_mean)
        feats_cv10_std.append(feats_frames_cv10_std)
        feats_cv30_mean.append(feats_frames_cv30_mean)
        feats_cv30_std.append(feats_frames_cv30_std)
        feats_swin_mean.append(feats_frames_swin_mean)
        feats_swin_std.append(feats_frames_swin_std)

    scio.savemat(os.path.join(out_path, out_name+'_ConvNext10_mean_feats.mat'), {'feats_mat':feats_cv10_mean})
    scio.savemat(os.path.join(out_path, out_name+'_ConvNext30_mean_feats.mat'), {'feats_mat':feats_cv30_mean})
    scio.savemat(os.path.join(out_path, out_name+'_SwinTrans_mean_feats.mat'), {'feats_mat':feats_swin_mean})
    scio.savemat(os.path.join(out_path, out_name+'_ConvNext10_std_feats.mat'), {'feats_mat':feats_cv10_std})
    scio.savemat(os.path.join(out_path, out_name+'_ConvNext30_std_feats.mat'), {'feats_mat':feats_cv30_std})
    scio.savemat(os.path.join(out_path, out_name+'_SwinTrans_std_feats.mat'), {'feats_mat':feats_swin_std})


if __name__=='__main__':

    model_names = ['convnext_xlarge_384_in22ft1k',
                   'convnext_xlarge_384_in22ft1k',
                   'swin_large_patch4_window12_384_in22k']
    model_paths = ['./save_result/models/convnext_xlarge_384_in22ft1k_10.pth',
                   './save_result/models/convnext_xlarge_384_in22ft1k_30.pth',
                   './save_result/models/swin_large_patch4_window12_384_in22k_40.pth']

    video_path = r'../../data/C3-cropped/'
    meta_file = r'../../metadata/c3_crop_metadata_withmos.csv'
    out_path = r'../../DFGCfeatures/dfgc-1st'
    out_name = 'DFGC-C3_DFGC1st'

    device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
    models = get_models(model_names, model_paths, device)

    feats_all_videos(video_path, meta_file, device, out_path)

