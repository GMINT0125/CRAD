#split and make json file for CRAD

import os
import sys
import random

import pandas as pd 
import json

random.seed(42)

class CONFIG():
    CROP_PATH = '../data/cropped_data/' 
    DATA_PATH = '../data/raw_data/'
    JSON_PATH = '../data/crad/json/'
    TRAIN_JSON_PATH = '../data/crad/json/train.json'
    TEST_JSON_PATH = '../data/crad/json/test.json'

config = CONFIG()

def make_json(config):

    images = [f for f in os.listdir(config.CROP_PATH) if f.endswith('.jpg')] #image file name
    labels = [f for f in os.listdir(config.DATA_PATH) if f.endswith('.txt')] #txt 파일
    images = pd.DataFrame(images, columns=['image'])

    images['label'] = 0 # 0 == normal

    for txt in labels:
        with open(config.DATA_PATH + txt, 'r') as f:
            label = f.read().split()
            if len(label) == 10:
                images.loc[images['image'] == txt.replace('.txt', '.jpg'), 'label'] = 1 # 1 == abnormal

    os.makedirs(config.JSON_PATH, exist_ok = True)

    for i in range(len(images)):
        image_name = images.iloc[i]['image']
        label = images.iloc[i]['label']

        if label == 0:
            line = {'filename': image_name, 'label': 0, 'label_name': 'good', 'clsname': 'head'}
            if random.random() < 0.2: #0.2의 확률로 
                with open(config.TEST_JSON_PATH, 'a', encoding='utf-8') as file:
                    file.write(json.dumps(line) + '\n')
            else:
                with open(config.TRAIN_JSON_PATH, 'a', encoding='utf-8') as file:
                    file.write(json.dumps(line) + '\n')
   
        else:
            line = {'filename': image_name, 'label': 1, 'label_name': 'abnormal', 'clsname': 'head'}
            with open(config.TEST_JSON_PATH, 'a', encoding='utf-8') as file:
                file.write(json.dumps(line) + '\n')

if __name__ == '__main__':
    make_json(config)