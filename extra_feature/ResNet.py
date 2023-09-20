from logging import raiseExceptions
import sys
import os
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import pandas as pd
from tqdm import tqdm
import scipy.io as scio

class VggExtractor(nn.Module):
    def __init__(self, train=False):
        super(VggExtractor, self).__init__()

        self.vgg = models.vgg19(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:4])

        if train:
            pass
        else:
            self.vgg.eval()

    def freeze_gradient(self, begin=0, end=9):
        for i in range(begin, end+1):
            self.vgg[i].requires_grad = False

    def forward(self, input):
        return self.vgg(input)

class ResNetExtractor(nn.Module):
    def __init__(self, train=False):
        super(ResNetExtractor, self).__init__()

        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        if train:
            pass
        else:
            self.resnet.eval()

    def forward(self, input):
        return self.resnet(input)

if __name__ == '__main__':
    model = 'ResNet50'
    device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')
    video_path = r'../../data/C3-cropped'
    meta_file = r'../../metadata/c3_crop_metadata_withmos.csv'
    out_path = r'../../features/ResNet'
    out_name = 'DFGC-C3_ResNet50'

    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(meta_file)
    framerate = list(df['framerate'])
    video_list = list(df['name'])

    if model == 'VGG19':
        ext = VggExtractor()
    elif model == 'ResNet50':
        ext = ResNetExtractor()

    ext.to(device)

    feats_mat_mean = []
    feats_mat_std = []
    with torch.no_grad():
        for i in tqdm(range(len(video_list))):
            #print('processing %s...'%video_list[i])
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
                    frame_tensor = Variable(torch.unsqueeze(frame_tensor, dim=0).float(), requires_grad=False)
                    frame_tensor = frame_tensor.to(device)
                    x = ext(frame_tensor)
                    x = x.squeeze()
                    feats_frames.append(x)

            feats_frames = torch.stack(feats_frames,dim=0)
            #out_file = os.path.join(out_path, str(video_list[i]+'.npy'))
            #np.save(out_file, feats_frames.cpu().numpy())
            feats_frames_mean = feats_frames.mean(0)
            feats_frames_std = feats_frames.std(0)
            feats_mat_mean.append(feats_frames_mean.cpu().numpy())
            feats_mat_std.append(feats_frames_std.cpu().numpy())

    feats_mat_mean = np.vstack(feats_mat_mean)
    feats_mat_std = np.vstack(feats_mat_std)
    
    out_mean = os.path.join(out_path, out_name+'_mean_feats.mat')
    out_std = os.path.join(out_path, out_name+'_std_feats.mat')
    scio.savemat(out_mean, {'feats_mat':feats_mat_mean})
    scio.savemat(out_std, {'feats_mat':feats_mat_std})


