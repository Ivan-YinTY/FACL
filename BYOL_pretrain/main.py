import os
import torch
from models.models import ResNet18
from models.models import MLPHead
from trainer import BYOLTrainer
import numpy as np
import random
from data.dataset import get_transforms, MultiViewDataInjector,VeinDataset
from torch.utils.data.dataloader import DataLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FVUSM')
    parser.add_argument('--network', type=str, default='resnet18')
    parser.add_argument('--loss', type=str, default='BYOL')
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--encoder_momentum', type=float, default=0.996, help='target encoder momentum')
    parser.add_argument('--wd', type=float, default=4e-4, help='weight decay')
    parser.add_argument('--max_epoch', type=int, default=80, help='margin')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=99, help='random seed for repeating results')
    parser.add_argument("--plot", action='store_true', help="plot roc and histogram")
    parser.add_argument('--synthetic_num', type=int, default=10000, help='the number of synthetic samples used for training')
    parser.add_argument('--traindata', type=str, default="", help='train set path')
    parser.add_argument("--multi_gpu", action='store_true', help="use multiple GPUs")
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    if args.dataset == "FVUSM":
        sample_per_class = 12
        test_path = "../vein_databases/FV-USM-processed"
        # test_path = "/home/weifeng/Desktop/Thesis_source_codes/chapter_5/vein_databases/FV-USM-processed"
    elif args.dataset == "Palmvein":
        sample_per_class = 20
        test_path = "../vein_databases/Palmvein_tongji"
        # test_path = "/home/weifeng/Desktop/Thesis_source_codes/chapter_5/vein_databases/Palmvein_tongji"
    else:
        ValueError("Dataset not supported!")

    data_transform_train, data_transform_test = get_transforms(args.dataset)
    train_path = args.traindata
    train_dataset = VeinDataset(root=train_path, transforms=MultiViewDataInjector([data_transform_train, data_transform_train]), num_sample=args.synthetic_num)
    # train_dataset = VeinDataset(root=train_path, sample_per_class=sample_per_class, transforms=MultiViewDataInjector([data_transform_train, data_transform_train]),
    #                            split='first_half')
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False, shuffle=True, pin_memory=True)

    test_dataset = VeinDataset(root=test_path, sample_per_class=sample_per_class, transforms=data_transform_test, split='second_half')
    testloader = DataLoader(test_dataset, batch_size=64, num_workers=4, drop_last=False, shuffle=False, pin_memory=True)

    # online network
    online_network = ResNet18(name=args.network).to(device)
    # predictor network
    predictor = MLPHead(in_channels=128, mlp_hidden_size=512, projection_size=128).to(device)
    # target encoder
    target_network = ResNet18(name=args.network).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    # optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=args.lr_decay)
    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          m=args.encoder_momentum,
                          max_epochs=args.max_epoch,
                          lr_scheduler=lr_scheduler,
                          args=args)

    trainer.train(trainloader, testloader)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    set_seed(args.seed)
    main()
