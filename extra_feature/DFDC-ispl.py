### code based on the original jupyter notebook for video prediction ###

import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms
from scipy.special import expit
from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet,weights
from isplutils import utils
import pandas as pd
import os, time
import cv2
from PIL import Image
import numpy as np
import scipy.io as scio
from tqdm import tqdm



if __name__=='__main__':

    net_model = 'EfficientNetAutoAttB4'
    train_db = 'DFDC'

    device = torch.device('cuda:8') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    
    model_url = weights.weight_url['{:s}_{:s}'.format(net_model,train_db)]
    net = getattr(fornet,net_model)().eval().to(device)
    net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

    video_path = r'../../C3-cropped'
    meta_file = r'../../metadata/c3_crop_metadata_withmos.csv'
    out_path = r'../../features/dfdc-ispl'
    out_name = 'DFGC-C3_DFDCispl'

    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(meta_file)
    video_list = list(df['name'])
    framerate = list(df['framerate'])
    
    videoreader = VideoReader(verbose=True)
    frame_num = 30

    feats_mat_mean = []
    feats_mat_std = []
    with torch.no_grad():
        for i in tqdm(range(len(video_list))):
            torch.cuda.empty_cache()

            video_file = os.path.join(video_path, video_list[i]+'.mp4') 
            camera = cv2.VideoCapture(video_file)

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
                    frame_Image = transforms.Resize([224,224])(frame_Image)
                    frame_tensor = transforms.ToTensor()(frame_Image)
                    frame_tensor = Variable(torch.unsqueeze(frame_tensor, dim=0).float(), requires_grad=False)
                    frame_tensor = frame_tensor.to(device)
                    x = net.features(frame_tensor).squeeze()
                    feats_frames.append(x)

            feats_frames = torch.stack(feats_frames,dim=0)
            feats_frames_mean = feats_frames.mean(0)
            feats_frames_std = feats_frames.std(0)
            #feats_frames_diff = torch.abs(feats_frames[1:,:]-feats_frames[:-1,:]).mean(0)
            #feats_frames = torch.cat((feats_frames_mean, feats_frames_std), dim=-1)
            feats_mat_mean.append(feats_frames_mean.cpu().numpy())
            feats_mat_std.append(feats_frames_std.cpu().numpy())

            '''
            # randomly extract frames
            frames, indices = videoreader.read_random_frames(path=video_file, num_frames=frame_num, seed=229)
            feats_frames = []
                                
            for frame in frames:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame_Image = Image.fromarray(frame)
                frame_Image = transforms.Resize(224)(frame_Image)
                frame_tensor = transforms.ToTensor()(frame_Image)
                frame_tensor = Variable(torch.unsqueeze(frame_tensor, dim=0), requires_grad=False)
                frame_tensor = frame_tensor.to(device)
                x = net.features(frame_tensor).squeeze()
                feats_frames.append(x)
                
            feats_frames = torch.stack(feats_frames,dim=0).mean(0)
            feats_mat.append(feats_frames.cpu().numpy())
            print('%s finished in %f seconds' %(video_list[i], time.time()-t0))
            '''
    
    feats_mat_mean = np.vstack(feats_mat_mean)
    feats_mat_std = np.vstack(feats_mat_std)
    scio.savemat(os.path.join(out_path, out_name+'_mean_feats.mat'), {'feats_mat':feats_mat_mean})
    scio.savemat(os.path.join(out_path, out_name+'_std_feats.mat'), {'feats_mat':feats_mat_std})
    