import os
import numpy as np
import torchvision.datasets
from torchvision import transforms
from PIL import Image

def getFVStatisticInfo(path, sample_number):

    img_h, img_w = 64, 128
    imgs = np.zeros([3, img_h, img_w, 1])
    means, stdevs = [], []
    data_transforms = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    path = os.path.abspath(path)
    for img_name in os.listdir(path):

        try:
            img = Image.open(os.path.join(path, img_name)).convert('RGB')

            img = data_transforms(img)
            img = img.numpy()

            img = img[:,:,:, np.newaxis]
            imgs =np.concatenate((imgs, img), axis= 3)

        except (OSError, NameError):
            print('OSError')

        for i in range(3):
            pixels = imgs[i, :, :, :].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        sample_number = sample_number - 1
        if (sample_number <= 0):
            break

    means = np.mean(means)
    stdevs = np.mean(stdevs)

    print('means is %f, std is %f'%(means, stdevs))

def getPytorchStatisticInfo(sample_number):
    train_data = torchvision.datasets.EuroSAT(
        root='D:/论文相关/MSc_Dissertation/Torch Proj/PyTorch_Tudui/dataset',
        # train = True,
        transform = torchvision.transforms.ToTensor(),
        download= True
    )

    img_h, img_w = 64, 64
    imgs = np.zeros([3, img_h, img_w, 1])
    means, stdevs = [], []

    for img_idx, _ in train_data:

        try:
            img = img_idx.numpy()

            img = img[:,:,:, np.newaxis]
            imgs =np.concatenate((imgs, img), axis= 3)

        except (OSError, NameError):
            print('OSError')

        for i in range(3):
            pixels = imgs[i, :, :, :].ravel()
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))

        sample_number = sample_number - 1
        if (sample_number <= 0):
            break

    means = np.mean(means)
    stdevs = np.mean(stdevs)

    print('means is %f, std is %f'%(means, stdevs))

if __name__ == '__main__':
    getFVStatisticInfo('D:/论文相关/MSc_Dissertation/Torch Proj/chapter_5/vein_databases/FV-USM-processed', sample_number = 500)
    #真实FV means is 0.251267, std is 0.119475
    #FV StyleGAN means is 0.220860, std is 0.116858
    #FV全局直方图均衡化 means is 0.503042, std is 0.289925
    #FV伽马变换 means is 0.523873, std is 0.121586
    #FV对数变换 means is 0.640062, std is 0.104809
    #FV StyleGAN-ADA means is 0.270897, std is 0.141280

    # getPytorchStatisticInfo(sample_number = 500)
    #CIFAR-10 32x32 means is 0.458815, std is 0.249357
    #CIFAR-100 32x32 means is 0.480851, std is 0.268609
    #EuroSAT 64x64 means is 0.471881, std is 0.158977
    #ImageNet2012 256x256 mean：[0.485, 0.456, 0.406],0.449,std：[0.229, 0.224, 0.225],0.226