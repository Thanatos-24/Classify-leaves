import torch
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os
import cv2
import json

def preprocess_image_label(folder_path, sigmaX=10):
    images = []
    # 获取文件夹中的所有图片文件路径
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("processing")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image not found at {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        # 移除过度的锐化处理，只进行基本的预处理
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
        image = image.astype(np.float32) / 255.0  # 归一化到[0,1]
        image = np.transpose(image, (2, 0, 1))
        images.append(image)
    return images

def preprocess_image_label_for_test(folder_path, sigmaX=10):
    images = []
    # 获取文件夹中的所有图片文件路径
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("processing")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image not found at {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        # 移除过度的锐化处理
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
        image = image.astype(np.float32) / 255.0  # 归一化到[0,1]
        image = np.transpose(image, (2, 0, 1))
        images.append((image, image_path))  # 返回路径
    return images

def preprocess_label(label_path, mapping_path="label_mapping.json"):
    label = pd.read_csv(label_path)
    label = label.iloc[:, 1].values
    unique_labels = sorted(set(label))  # 获取所有唯一的标签
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)} 
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f)
    print(f"Label mapping saved to {mapping_path}") # 映射为整数
    label = torch.tensor([label_mapping[l] for l in label])
    label = label.view(-1)
    return label.long()

if __name__ == "__main__":
    folder_path = "/Users/wangzhou/leaves/data/train"
    label_path = "/Users/wangzhou/leaves/data/train.csv"
    images = preprocess_image_label(folder_path)
    label = preprocess_label(label_path)
    print(images[1].shape)
    print(label[0])