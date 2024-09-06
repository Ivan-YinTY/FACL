import os
import cv2
from PIL import Image
import numpy as np
import glob
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader


class VeinDataset(data.Dataset):
    def __init__(self, root, sample_per_class=1, transforms=None, split='all', inter_aug='', num_sample=None):
        self.transform = transforms
        self.files = sorted(glob.glob(os.path.join(root) + "/*.bmp"))
        if num_sample is not None:
            self.files = sorted(glob.glob(os.path.join(root) + "/*.bmp"))[:num_sample]
        self.img_data = []

        if split == 'first_half' or split == 'second_half':
            self.class_num = len(self.files) // sample_per_class // 2
            self.labels = np.arange(self.class_num).repeat(sample_per_class)
        elif split == 'all':
            self.class_num = len(self.files) // sample_per_class
            self.labels = np.arange(self.class_num).repeat(sample_per_class)

        if split == 'first_half':
            # self.img_data = [cv2.imread(os.path.join(root, self.files[i])) for i in np.arange(0, len(self.files) // 2)]
            self.img_data = [cv2.imread(os.path.join(self.files[i])) for i in np.arange(0, len(self.files) // 2)]

        elif split == 'second_half':
            # self.img_data = [cv2.imread(os.path.join(root, self.files[i])) for i in np.arange(len(self.files) // 2, len(self.files))]
            self.img_data = [cv2.imread(os.path.join(self.files[i])) for i in np.arange(len(self.files) // 2, len(self.files))]

        elif split == 'all':
            # self.img_data = [cv2.imread(os.path.join(root, self.files[i])) for i in np.arange(0, len(self.files))]
            self.img_data = [cv2.imread(os.path.join(self.files[i])) for i in np.arange(0, len(self.files))]


        if inter_aug == 'LR':  # horizontal flip
            # self.img_data.extend([self.img_data[i].transpose(Image.FLIP_LEFT_RIGHT) for i in np.arange(0, len(self.img_data))])
            self.img_data.extend([self.img_data[i][:, ::-1, :] for i in np.arange(0, len(self.img_data))])
            aug_classes = np.arange(self.class_num, self.class_num * 2).repeat(sample_per_class)
            self.labels = np.concatenate([self.labels, aug_classes])
            self.class_num = self.class_num * 2
        elif inter_aug == 'TB':  # Vertical flip
            # self.img_data.extend([self.img_data[i].transpose(Image.FLIP_TOP_BOTTOM) for i in np.arange(0, len(self.img_data))])
            self.img_data.extend([self.img_data[i][::-1, :, :] for i in np.arange(0, len(self.img_data))])
            aug_classes = np.arange(self.class_num, self.class_num * 2).repeat(sample_per_class)
            self.labels = np.concatenate([self.labels, aug_classes])
            self.class_num = self.class_num * 2
        print(len(self.labels))
        print(len(self.img_data))
        pass

    def __getitem__(self, index):
        image = Image.fromarray(self.img_data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.img_data)


def get_transforms(dataset):
    # normalize to [-1, 1]
    normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    transform_train = []
    if dataset == 'Palmvein':
        transform_train.append(transforms.ColorJitter(brightness=0.9, contrast=0.9))
    elif dataset == 'FVUSM':
        transform_train.append(transforms.RandomResizedCrop(size=(64, 128), scale=(0.5, 1.0), ratio=(1.5, 2.5)))
        transform_train.append(transforms.RandomRotation(degrees=3))
        transform_train.append(transforms.ColorJitter(brightness=0.7, contrast=0.7))
    else:
        ValueError("Dataset not exist!")

    transform_train.append(transforms.ToTensor())
    transform_train.append(normalize)
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    return transform_train, transform_test


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]

    def __call__(self, sample):
        output = [transform(sample) for transform in self.transforms]
        return output