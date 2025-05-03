import pandas as pd
import os 
import sys
from tqdm import tqdm

import cv2
from ultralytics import YOLO
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
device

class CONFIG():
    RAW_DATAPATH = '../data/raw_data/'
    YOLO_PARAM = '../ultralytics/runs/detect/tune-n/weights/best.pt' # <- param 경로 수정
    CROP_DATAPATH = '../data/cropped_data/'

config = CONFIG()

def crop_img(config):
    model = YOLO(config.YOLO_PARAM) #가중치 로드
    dataset = os.listdir(config.RAW_DATAPATH)

    image = []
    image_name = []

    for data in dataset:
        if data.endswith('jpg'):
            image_name.append(data)
            raw_data = config.RAW_DATAPATH + data
            image.append(raw_data)

    print('head detection...')
    results = model.predict(image, device = device, classes = [80]) #head 탐지

    #box 좌표 저장
    box = []
    for i in range(len(results)):
        box_coord = results[i].boxes.xyxy.to('cpu').squeeze()
        box.append(box_coord)

    #DataFrame
    df = pd.DataFrame(image, columns = ['image'])
    df['xyxy'] = box


    for i in tqdm(range(len(df))):

        os.makedirs(config.CROP_DATAPATH, exist_ok = True)

        #Image cropping
        image_path = df.iloc[i]['image']
        xyxy = df.iloc[i]['xyxy']
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        crop_img = img[int(xyxy[3]):, int(xyxy[0]):int(xyxy[2])]
         

        save_path = config.CROP_DATAPATH + image_name[i]
        cv2.imwrite(save_path ,crop_img)


if __name__ == '__main__':
    crop_img(config)