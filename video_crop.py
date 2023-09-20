import cv2
import os
import threading
import pandas as pd
import numpy as np
from ffmpy import FFmpeg

###### extract frames from a single video
def extract_frame_single(video_path, tmp_path, name, frequency):

    times = 0
    pic_path=[]
         
    # read frames
    camera = cv2.VideoCapture(video_path)
     
    while True:
        times = times + 1
        res, image = camera.read()
        if not res:
            break
        if times % frequency == 0:
            pic_path_single = str(tmp_path + r'/' + name +'-'+ str(times)+'.png')
            cv2.imwrite(pic_path_single, image)
            pic_path.append(pic_path_single)
            
    camera.release()
    print("frame extraction for video %s has been finished" % name)
    return pic_path


###### detect face in a single picture, there should only be 1 face per pic
###### return:[x_center, y_center, w, h]
def detact_face_single(pic_path):

    img = cv2.imread(pic_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_ori = float(img.shape[1])
    y_ori = float(img.shape[0])
    
    ## face detection
    face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_alt2.xml')
    face_raw = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(200,200))  ## returns [x,y,w,h]

    os.remove(pic_path)
    
    if len(face_raw)!=1:
        face = np.array([-1, -1, -1, -1])
        return face
    else:
        for x,y,w,h in face_raw:
            face = np.array([(x+w/2)/x_ori, (y+h/2)/y_ori, w/x_ori, h/y_ori])
            return face

def crop_single(video_path, faces, name, width, height, expand_rate, output_path):
    
    location = np.mean(faces[:,:2], axis=0)
    shape = np.max(faces[:, 2:4], axis=0)

    # zoom bbox according to expand_rate
    w = float(shape[0]*width)*expand_rate
    h = float(shape[1]*height)*expand_rate
    x = max(int(location[0]*width-w/2),0)
    y = max(int(location[1]*height-h/2),0)
    if (x+w)>=width: w = int(width-x-1)
    if (y+h)>=height: h = int(height-y-1)
    print('crop prams: ',w,h,x,y,width,height)
        
    output = os.path.join(output_path,str(name+'-crop.mp4'))
    ff = FFmpeg(inputs={video_path: None},
                outputs={output: '-vf crop={}:{}:{}:{} -y -loglevel level+warning -preset ultrafast -strict -2'.format(int(w),int(h),x,y)})
    ff.run()
    print("crop for video %s has been finished" % name)


def process_single(video_path, tmp_path, output_path, name, width, height, frequency, expand_rate):
    
    pic_path_group = extract_frame_single(video_path, tmp_path, name, frequency)
    
    faces = []
    for pic_path in pic_path_group:
        face = detact_face_single(pic_path)
        if face[0]==-1: pass
        else: faces.append(face)

    faces = np.squeeze(np.array(faces))
    print(faces.shape)
    try:
        if faces.shape[1]==4:
            print("face detection for video %s has been finished" % name)
            crop_single(video_path, faces, name, width, height, expand_rate, output_path)
    except:
        print("!!!face detection for video %s has been failed !!!" % name)


###### detect & crop accroding to each video ######
def process(source_path, metadata_path, tmp_path, output_path, extract_rate, expand_rate):
    
    if not os.path.exists(tmp_path): os.makedirs(tmp_path)

    count = 0

    video_data = pd.read_csv(metadata_path)
    names = list(video_data.name)
    frame_rates = list(video_data.framerate)
    width = list(video_data.width)
    height = list(video_data.height)

    for name in names:
        video_path = os.path.join(source_path, name+'.mp4')
        frequency = int(frame_rates[count]/extract_rate)
        threading.Thread(target=process_single, args=(video_path, tmp_path, output_path, name, width[count], height[count], frequency, expand_rate)).start()
        count = count + 1
        #if count==1:break   



def process_batch_single(source_path, metadata_path, tmp_path, output_path, name, frequency, expand_rate, submit_id):
    
    video_data = pd.read_csv(metadata_path)
    names = list(video_data.name)
    
    ### get bbox from target video
    target_path = os.path.join(source_path, (name+'.mp4'))
    target_pic_path = extract_frame_single(target_path, tmp_path, name, frequency)

    faces = []
    for pic_path in target_pic_path:
        face = detact_face_single(pic_path)
        if face[0]==-1: pass
        else: faces.append(face)

    faces = np.squeeze(np.array(faces))
    print(faces.shape)

    ### crop the whole batch
    try:
        if faces.shape[1]==4:
            print("face detection for video %s has been finished" % name)
            for submit in submit_id:
                video_name = str(name.replace('target','submit-')+submit)
                video_path = os.path.join(source_path, str(video_name+'.mp4'))
                if not os.path.exists(video_path):print("!!! wrong submit id:", submit)
                else:
                    index = names.index(video_name)
                    width = video_data.width[index]
                    height = video_data.height[index]
                    crop_single(video_path, faces, video_name, width, height, expand_rate, output_path)
    except:
        print("!!!face detection for video %s has been failed !!!" % name)




###### detect & crop accroding to each 'target' video ######
def process_batch(source_path, metadata_path, tmp_path, output_path, extract_rate, expand_rate, submit_id):
    if not os.path.exists(tmp_path): os.makedirs(tmp_path)
    if not os.path.exists(output_path): os.makedirs(output_path)
    
    count = 0
    count_target = 0

    video_data = pd.read_csv(metadata_path)
    names = list(video_data.name)
    frame_rates = list(video_data.framerate)
    width = list(video_data.width)
    height = list(video_data.height)

    for name in names:
        if name[-6:]=='target':
            frequency = int(frame_rates[count]/extract_rate)
            threading.Thread(target=process_batch_single, args=(source_path, metadata_path, tmp_path, output_path, name, frequency, expand_rate, submit_id)).start()
            count_target = count_target + 1
        else:pass
        count = count + 1


###### main ######
if __name__=='__main__':

    phase = ['C1', 'C2', 'C3']
    extract_rate = 4 # number of frames to be extracted per second
    expand_rate = 1.3 # expand the detected box 1.3 times to include the whole head.
    
    for p in phase:
        print(f'processing phase {p} ...')

        source_path = r'./data/'+p
        metadata_path = r'./metadata/DFGC2022_'+p+'_metadata_full.csv'
        tmp_path = r'./cropped/tmp'
        output_path = r'./data/'+p+'-cropped' ## output dir

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        if p =='C1':
            submit_id = ['00000', '73479', '73549', '74734', '74766', '75090']  # for C1
        elif p == 'C2':
            submit_id = ['00000', '82495', '82501', '82508', '82511', '82882', '82910', '82999', '83063', '83485', '83496', '83609', '83619']  # for C2
        elif p == 'C3': 
            submit_id = ['00000', '91740', '92068', '92069', '92147', '92582', '92584', '93014', '93056', '93059', '93060', '93062', '93065', '93110', '93169', '93170']  # for C3

        process_batch(source_path, metadata_path, tmp_path, output_path, extract_rate, expand_rate, submit_id)
    

