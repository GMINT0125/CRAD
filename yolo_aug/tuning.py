#tuning hyperparameter for YOLO

import os
import torch
import torchvision

from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def tuning():
    
    model = YOLO("yolo11n.pt") 
    model.tune(data="coco8.yaml", epochs = 100, iterations = 100, workers=1, device = 0, batch=16)

if __name__ == "__main__":
    tuning()