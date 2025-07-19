import torch
from torch.utils.data import Dataset
from dataloader import preprocess_image_label, preprocess_label
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

class train_valid_dataset(Dataset):
    def __init__(self, data_list, data_label, mode='train'):
        self.data_list = data_list
        self.data_label = data_label
        self.mode = mode
        self.start_index = 0
        self.end_index = 0
        
        if mode == 'train':
            self.start_index = 0
            self.end_index = int(len(data_label) * 0.8)
        elif mode == 'valid':
            self.start_index = int(len(data_label) * 0.8)
            self.end_index = int(len(data_label))
        
        self.data = data_list[self.start_index:self.end_index]
        self.label = data_label[self.start_index:self.end_index]
        
        # 定义数据增强和标准化
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        
        # 将数据从(3, 224, 224)转换为(224, 224, 3)格式，以便ToPILImage能正确处理
        if len(data.shape) == 3 and data.shape[0] == 3:
            data = np.transpose(data, (1, 2, 0))
        
        # 确保数据类型正确 - ToPILImage需要uint8格式
        if data.dtype == np.float32:
            data = (data * 255).astype(np.uint8)
        
        # 应用数据增强和标准化
        data = self.transform(data)
        
        return data, label

class test_dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list  # List of (img_array, img_path)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_array, img_path = self.data_list[index]
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        return img_tensor, img_path  # 返回 tuple，供 DataLoader 解包

if __name__ == "__main__":
    folder_path = "/Users/wangzhou/leaves/data/images"
    label_path = "/Users/wangzhou/leaves/data/train.csv"
    
    images = preprocess_image_label(folder_path)
    labels = preprocess_label(label_path)
    
    # Convert images and labels to numpy arrays
    images = np.array(images)
    labels = labels.numpy()
    
    # Create datasets
    train_dataset = train_valid_dataset(images, labels, mode='train')
    valid_dataset = train_valid_dataset(images, labels, mode='valid')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    
    # Example of accessing data
    data, label = train_dataset[0]
    print(data.shape, label.shape)
