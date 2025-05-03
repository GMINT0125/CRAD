#train valid split for YOLO

import os
import sys

import random
import pandas as pd 
import numpy as np
import shutil

from sklearn.model_selection import train_test_split


random.seed(42)

class CONFIG():
    DATA_PATH = '../data/raw_data/'
    TRAIN_IMAGE = '../data/images/train/'
    TRAIN_LABEL = 'data/labels/train/'
    VAL_IMAGE = '../data/images/val/'
    VAL_LABEL = '../data/labels/val/' 
    
config = CONFIG()

os.makedirs(config.TRAIN_IMAGE, exist_ok= True)
os.makedirs(config.TRAIN_LABEL, exist_ok= True)
os.makedirs(config.VAL_IMAGE, exist_ok= True)
os.makedirs(config.VAL_LABEL, exist_ok= True)

def split(config, ratio):

    images = [f for f in os.listdir(config.DATA_PATH) if f.endswith('.jpg')]
    labels = [f for f in os.listdir(config.DATA_PATH) if f.endswith('.txt')] #txt 파일
    images = pd.DataFrame(images, columns=['image'])

    images['label'] = 0 # 0 == normal

    for txt in labels:
        with open(config.DATA_PATH + txt, 'r') as f:
            label = f.read().split()
            if len(label) == 10:
                images.loc[images['image'] == txt.replace('.txt', '.jpg'), 'label'] = 1 # 1 == abnormal



    train, valid = train_test_split(images, test_size = ratio, stratify = images['label']) #stratify: label의 비율을 유지
    print(len(train), len(valid))

    for train_image in train['image']:
        shutil.copy(config.DATA_PATH + train_image, config.TRAIN_IMAGE + train_image) #jpg 파일 복사
        shutil.copy(config.DATA_PATH + train_image.replace('.jpg', '.txt'), config.TRAIN_LABEL + train_image.replace('.jpg', '.txt'))

    for valid_image in valid['image']:
        shutil.copy(config.DATA_PATH + valid_image, config.VAL_IMAGE + valid_image)
        shutil.copy(config.DATA_PATH + valid_image.replace('.jpg', '.txt'), config.VAL_LABEL + valid_image.replace('.jpg', '.txt'))



if __name__ == '__main__':
    split(config, ratio = 0.2)