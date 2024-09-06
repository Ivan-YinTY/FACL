import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from datasets.datasets import FingerVeinDataset
from utils import utils
import argparse
import numpy as np
import os
from evaluation.evaluation import evaluate_verification
from networks import resnet
from torchvision.utils import save_image
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FVUSM')
    parser.add_argument('--network', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--max_epoch', type=int, default=80, help='margin')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=99, help='random seed for repeating results')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument("--pretrained", action='store_true', help="pretrained on imageNet")
    parser.add_argument("--multi_gpu", action='store_true', help="use multi gpu")
    parser.add_argument('--synthetic_num', type=int, default=6000, help='the number of synthetic samples used for training')
    parser.add_argument('--traindata', type=str, default="", help='train set path')
    parser.add_argument('--t', type=float, default=0.04, help='temperature')
    parser.add_argument('--loss', type=str, default="SimCLR")
    args = parser.parse_args()
    return args


def save_model(model, current_result, best_result, best_snapshot, lower_is_better=True):
    aver = current_result['aver']
    epoch = current_result['epoch']
    prefix = 'seed=%d_dataset=%s_network=%s_loss=%s' % (args.seed, args.dataset, args.network, args.loss)
    # save the current best model
    if best_result is None or (aver >= best_result['aver'] and not lower_is_better) \
            or (aver <= best_result['aver'] and lower_is_better):
        best_result = current_result
        snapshot = {'model': model.module.state_dict() if torch.cuda.device_count() > 1 and args.multi_gpu else model.state_dict(),
                    'epoch': epoch,
                    'args': args
                    }
        if best_snapshot is not None:
            os.system('rm %s' % (best_snapshot))

        best_snapshot = './snapshots/%s_Best%s=%.2f_Epoch=%d.pth' % (
            prefix, 'ROC' if lower_is_better else 'CMC', aver * 100, epoch)
        torch.save(snapshot, best_snapshot)
    # always save the final model
    if epoch == args.max_epoch - 1:
        snapshot = {'model': model.module.state_dict() if torch.cuda.device_count() > 1 and args.multi_gpu else model.state_dict(),
                    'epoch': epoch,
                    'args': args
                    }
        last_snapshot = './snapshots/%s_Final%s=%.2f_Epoch=%d.pth' % (
            prefix, 'ROC' if lower_is_better else 'CMC', aver * 100, epoch)
        torch.save(snapshot, last_snapshot)
    return best_result, best_snapshot


def get_transformer(dataset):
    normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    transform_train = []
    if dataset == "FVUSM":
        transform_train.append(transforms.RandomResizedCrop(size=(64, 128), scale=(0.5, 1.0), ratio=(1.5, 2.5)))
        transform_train.append(transforms.RandomRotation(degrees=3))
        transform_train.append(transforms.ColorJitter(brightness=0.7, contrast=0.7))
    elif dataset == "Palmvein":
        transform_train.append(transforms.ColorJitter(brightness=0.9, contrast=0.9))
    transform_train.append(transforms.ToTensor())
    transform_train.append(normalize)
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    return transform_train, transform_test


def train(model, trainloader, optimizer, epoch):
    model.train()
    loss_stats = utils.AverageMeter()
    for batch_idx, data in enumerate(trainloader):
        view_1, view_2 = data["view_1"].cuda(), data["view_2"].cuda()
        _, f_view_1 = model(view_1)
        _, f_view_2 = model(view_2)
        # show some augmented pairs
        sample = torch.cat((view_1, view_2), -1)
        save_image(sample[:10], "results/%d.bmp" % (batch_idx), nrow=1, normalize=True, range=(-1.0, 1.0))
        # contrastive loss
        f_view_1_nml = F.normalize(f_view_1, p=2, dim=1)
        f_view_2_nml = F.normalize(f_view_2, p=2, dim=1)
        p = f_view_1_nml.matmul(f_view_2_nml.T)
        loss_1 = -1 * F.log_softmax(p / args.t, dim=1).diag().mean()
        loss_2 = -1 * F.log_softmax(p / args.t, dim=0).diag().mean()
        loss = (loss_1 + loss_2) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_stats.update(loss.item())
        print(utils.dt(), 'Epoch:[%d]-[%d/%d] batchLoss:%.4f averLoss:%.4f' %
              (epoch, batch_idx, len(trainloader), loss_stats.val, loss_stats.avg))


def train_epochs_openset(model, trainloader, testloader, optimizer, lr_scheduler):
    max_epoch = args.max_epoch
    best_result = None
    best_snapshot = None
    print(utils.dt(), 'Training started.')
    for epoch in range(max_epoch):
        train(model, trainloader, optimizer, epoch)
        roc, aver, auc = evaluate_verification(model, testloader)
        lr_scheduler.step()
        # save the current best model based on eer
        best_result, best_snapshot = \
            save_model(model, {'metrics': roc,  'aver': roc[0], 'epoch': epoch}, best_result, best_snapshot)

    print(utils.dt(), 'Training completed.')
    print(utils.dt(), '------------------Best Results---------------------')
    epoch, roc = best_result['epoch'], best_result['metrics']
    print(utils.dt(), 'EER: %.2f%%, FPR100:%.2f%%, FPR1000:%.2f%%, FPR10000:%.2f%%, FPR0:%.2f%%, Aver: %.2f%% @ epoch %d' %
          (roc[0]*100, roc[1]*100, roc[2]*100, roc[3]*100, roc[4]*100, np.mean(roc)*100, epoch))


def main():
    transform_train, transform_test = get_transformer(args.dataset)
    trainset_path = args.traindata
    from datasets.datasets import ImageDataset
    trainset = ImageDataset(trainset_path, transform_train, num=args.synthetic_num)
    if args.dataset == 'FVUSM':
        testset_path = "../vein_databases/FVUSM-seg"
        test_list = './datasets/FVUSM_testlist_openset.txt'
    elif args.dataset == 'Palmvein':
        testset_path = "../vein_databases/Palmvein_tongji"
        test_list = './datasets/Palmvein_testlist.txt'
    else:
        raise ValueError('Dataset %s not exists!' % (args.dataset))

    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testset = FingerVeinDataset(filelist_path=test_list, dataset_path=testset_path, transform=transform_test)
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if args.network == 'resnet18':
        model = resnet.resnet18(pretrained=args.pretrained)
    elif args.network == 'resnet34':
        model = resnet.resnet34(pretrained=args.pretrained)
    elif args.network == 'resnet50':
        model = resnet.resnet50(pretrained=args.pretrained)
    else:
        raise ValueError('Network %s not supported!' % (args.network))

    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
        model_module = model.module
    else:
        model = model.cuda()
        model_module = model

    optimizer = torch.optim.SGD(model_module.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model_module.parameters(), lr=args.lr)  # palmvein
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=args.lr_decay)
    train_epochs_openset(model, trainloader, testloader, optimizer, lr_scheduler)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    utils.set_seed(args.seed)
    main()
