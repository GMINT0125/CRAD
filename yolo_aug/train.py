#train the YOLO

import os
import torch

from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def train():
    
    model = YOLO("yolo11x.pt") 
    results = model.train(data="coco8.yaml", epochs = 100, batch=16, workers=1, device = 0)
    return results

if __name__ == "__main__":
    train()

