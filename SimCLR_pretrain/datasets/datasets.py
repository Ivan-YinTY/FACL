import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import cv2
from PIL import Image
import numpy as np
import random
import glob


class ImageDataset_1(Dataset):
    def __init__(self, root, transform=None, class_num=None):
        self.transform = transform
        if class_num is not None:
            self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))[:class_num]
        else:
            self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))
        self.labels = torch.arange(len(self.files))
        self.class_num = len(set(self.labels))
        pass

    def __getitem__(self, index):
        image = cv2.imread(self.files[index])
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.files)


class ImageDataset(Dataset):
    def __init__(self, root, transform, num=None):
        self.transform = transform
        if num is not None:
            self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))[:num]
        else:
            self.files = sorted(glob.glob(os.path.join(root) + "/*.*"))
        pass

    def __getitem__(self, index):
        image = cv2.imread(self.files[index])
        image = Image.fromarray(image)
        image_1 = self.transform(image)
        image_2 = self.transform(image)
        return {"view_1": image_1, "view_2": image_2}

    def __len__(self):
        return len(self.files)


class FingerVeinDataset(Dataset):
    def __init__(self, filelist_path, dataset_path, transform=None, three_channel=True):
        filelist = open(filelist_path).readlines()
        self.img_paths = [line.split()[0] for line in filelist]
        self.labels = [int(line.split()[1]) for line in filelist]
        self.dataset_path = dataset_path
        self.transform = transform
        self.three_channel = three_channel
        self.class_num = len(set(self.labels))

    def __getitem__(self, item):
        if self.three_channel:
            image = cv2.imread(os.path.join(self.dataset_path, self.img_paths[item]))
            image = Image.fromarray(image)
        else:
            image = Image.open(os.path.join(self.dataset_path, self.img_paths[item]))
        target = self.labels[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.labels)


