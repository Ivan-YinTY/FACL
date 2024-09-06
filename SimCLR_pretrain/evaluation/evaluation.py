import numpy as np
import torch
from sklearn import preprocessing, metrics
from itertools import combinations
import matplotlib.pyplot as plt
from utils import utils
from torchvision import transforms
from time import time
import os
import numpy as np
from sklearn import manifold
import random
import sklearn


def get_embeddings(model, testloader):
    model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for data, target in testloader:
            data = data.cuda()
            f = model(data)
            f = f[0] if isinstance(f, tuple) else f
            embeddings.append(f.data.cpu().numpy())
            targets.append(target.data.cpu().numpy())
    embeddings = np.vstack(embeddings)
    embeddings = preprocessing.normalize(embeddings)
    targets = np.concatenate(targets)
    model.train()
    return embeddings, targets


def compute_roc(model, testloader):
    embeddings, targets = get_embeddings(model, testloader)
    emb_num = len(embeddings)
    # Cosine similarity between any two pairs, note that all embeddings are l2-normalized
    scores = np.matmul(embeddings, embeddings.T)
    class_num = testloader.dataset.class_num
    samples_per_class = emb_num // class_num
    # define matching pairs
    intra_class_combinations = np.array(list(combinations(range(samples_per_class), 2)))
    match_pairs = [i*samples_per_class + intra_class_combinations for i in range(class_num)]
    match_pairs = np.concatenate(match_pairs, axis=0)
    scores_match = scores[match_pairs[:, 0], match_pairs[:, 1]]
    labels_match = np.ones(len(match_pairs))

    # define imposter pairs
    inter_class_combinations = np.array(list(combinations(range(class_num), 2)))
    imposter_pairs = [np.expand_dims(i*samples_per_class, axis=0) for i in inter_class_combinations]
    imposter_pairs = np.concatenate(imposter_pairs, axis=0)
    scores_imposter = scores[imposter_pairs[:, 0], imposter_pairs[:, 1]]
    labels_imposter = np.zeros(len(imposter_pairs))

    # merge matching pairs and imposter pairs and assign labels
    all_scores = np.concatenate((scores_match, scores_imposter))
    all_labels = np.concatenate((labels_match, labels_imposter))
    # compute roc, auc and eer
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    return fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets


def compute_roc_two_session(model, testloader):
    # extract feature embeddings
    embeddings, targets = get_embeddings(model, testloader)
    emb_num = len(embeddings)
    # Cosine similarity between any two pairs, note that all embeddings are l2-normalized
    scores = np.matmul(embeddings, embeddings.T)
    class_num = testloader.dataset.class_num
    samples_per_class = emb_num // class_num

    ind_all = np.arange(0, emb_num)
    ind_session_1 = []
    ind_session_2 = []
    for i in range(0, emb_num, samples_per_class):
        ind_session_1.append(ind_all[i: i + samples_per_class // 2])
        ind_session_2.append(ind_all[i + samples_per_class // 2: i + samples_per_class])
    # define genuine pairs
    genuine_pairs = []
    for i in range(0, class_num):
        s1 = ind_session_1[i]
        s2 = ind_session_2[i]
        # genuine_pairs.extend([[x, y] for x in s1 for y in s2])
        genuine_pairs.append(np.array(np.meshgrid(s1, s2)).T.reshape(-1, 2))
    genuine_pairs = np.concatenate(genuine_pairs, 0)
    # genuine_pairs = np.array(genuine_pairs)
    genuine_scores = scores[genuine_pairs[:, 0], genuine_pairs[:, 1]]
    labels_genuine = np.ones(len(genuine_scores))
    print(len(genuine_scores))
    # define imposter pairs
    imposter_pairs = []
    for i in range(0, class_num):
        s1 = ind_session_1[i]
        ind_session_2_copy = ind_session_2.copy()
        ind_session_2_copy.pop(i)
        s2 = np.concatenate(ind_session_2_copy, 0)
        # imposter_pairs.extend([[x, y] for x in s1 for y in s2])
        imposter_pairs.append(np.array(np.meshgrid(s1, s2)).T.reshape(-1, 2))
    imposter_pairs = np.concatenate(imposter_pairs, 0)
    # imposter_pairs = np.array(imposter_pairs)
    imposter_scores = scores[imposter_pairs[:, 0], imposter_pairs[:, 1]]
    labels_imposter = np.zeros(len(imposter_pairs))
    print(len(imposter_scores))
    # merge matching pairs and imposter pairs and assign labels
    all_scores = np.concatenate((genuine_scores, imposter_scores))
    all_labels = np.concatenate((labels_genuine, labels_imposter))
    # compute roc, auc and eer
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    return fpr, tpr, thresholds, genuine_scores, imposter_scores, embeddings, targets


def compute_roc_metrics(fpr, tpr, thresholds):
    fnr = 1 - tpr
    # find indices where EER, fpr100, fpr1000, fpr0, best acc occur
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    fpr100_idx = sum(fpr <= 0.01) - 1
    fpr1000_idx = sum(fpr <= 0.001) - 1
    fpr10000_idx = sum(fpr <= 0.0001) - 1
    fpr0_idx = sum(fpr <= 0.0) - 1

    # compute EER, FRR@FAR=0.01, FRR@FAR=0.001, FRR@FAR=0
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    fpr100 = fnr[fpr100_idx]
    fpr1000 = fnr[fpr1000_idx]
    fpr10000 = fnr[fpr10000_idx]
    fpr0 = fnr[fpr0_idx]

    metrics = (eer, fpr100, fpr1000, fpr10000, fpr0)
    metrics_thred = (thresholds[eer_idx], thresholds[fpr100_idx], thresholds[fpr1000_idx], thresholds[fpr10000_idx], thresholds[fpr0_idx])
    AUC = sklearn.metrics.auc(fpr, tpr)
    print(utils.dt(), 'Performance evaluation...')
    print('EER:%.2f%%, FRR@FAR=0.01: %.2f%%, FRR@FAR=0.001: %.2f%%, FRR@FAR=0.0001: %.2f%%, FRR@FAR=0: %.2f%%, Aver: %.2f%%, AUC:%.2f%%' %
          (eer * 100, fpr100 * 100, fpr1000 * 100, fpr10000 * 100, fpr0 * 100, np.mean(metrics) * 100, AUC * 100))
    return metrics, metrics_thred, AUC


def evaluate_verification(model, testloader):
    fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets = compute_roc_two_session(model, testloader)
    roc_metrics, metrics_threds, AUC = compute_roc_metrics(fpr, tpr, thresholds)
    return roc_metrics, np.mean(roc_metrics), AUC


def compute_roc_from_snapshot(model_path):
    from datasets.datasets import FingerVeinDataset
    from torch.utils.data.dataloader import DataLoader

    snapshot = torch.load(model_path)
    args = snapshot['args']
    if args.dataset == 'FVUSM':
        testset_path = "/home/weifeng/Desktop/FV-USM-processed"
        test_list = '../datasets/FVUSM_testlist_openset.txt'
    elif args.dataset == 'Palmvein':
        testset_path = "/home/weifeng/Desktop/Palmvein_tongji"
        test_list = '../datasets/Palmvein_testlist.txt'
    else:
        raise ValueError('Dataset %s not exists!' % (args.dataset))

    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    testset = FingerVeinDataset(filelist_path=test_list, dataset_path=testset_path, transform=transform_test)
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    from networks.resnet import resnet18, resnet34, resnet50
    if args.network == 'resnet18':
        model = resnet18()
    elif args.network == 'resnet34':
        model = resnet34()
    elif args.network == 'resnet50':
        model = resnet50()
    else:
        raise ValueError('Network %s not supported!' % (args.network))

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in snapshot['model'].items() if k in model_dict and k != 'prelu_fc1.weight' and k != 'fc1.weight' and k != 'fc1.bias' and k != 'fc2.weight' and k != 'fc2.bias'}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = torch.nn.DataParallel(model).cuda()

    # fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets = compute_roc(model, testloader)
    fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets = compute_roc_two_session(model, testloader)
    roc_metrics, metrics_threds, auc = compute_roc_metrics(fpr, tpr, thresholds)
    return fpr, tpr, thresholds, scores_match, scores_imposter


def analyze_pair_distance_new(model_path):
    from datasets.datasets import FingerVeinDataset
    from networks.resnet import resnet18, resnet34, resnet50
    from torch.utils.data.dataloader import DataLoader

    snapshot = torch.load(model_path)
    args = snapshot['args']

    real_path = "/home/weifeng/Desktop/FV-USM-processed"
    real_testlist = '../datasets/FVUSM_testlist_openset.txt'

    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    from datasets.datasets import ImageDataset_1
    stylegan_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/stylegan2-pytorch-master/fv_samples_150ep"
    stylegan_loader = DataLoader(dataset=ImageDataset_1(root=stylegan_path, transform=transform_test), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    real_testloader = DataLoader(dataset=FingerVeinDataset(filelist_path=real_testlist, dataset_path=real_path, transform=transform_test, three_channel=True), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if args.network == 'resnet18':
        model = resnet18()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in snapshot['model'].items() if
                           k in model_dict and k != 'prelu_fc1.weight' and k != 'fc1.weight' and k != 'fc1.bias' and k != 'fc2.weight' and k != 'fc2.bias'}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = model.cuda()

    embeddings_stylegan, _ = get_embeddings(model, stylegan_loader)
    embeddings_realtest, _ = get_embeddings(model, real_testloader)
    num = 10000
    scores_sty_sty = np.matmul(embeddings_stylegan[:num], embeddings_stylegan[:num].T)
    scores_sty_sty = scores_sty_sty[np.triu_indices(len(scores_sty_sty), 1)]
    print(len(scores_sty_sty))

    scores_real_all = np.matmul(embeddings_realtest, embeddings_realtest.T)
    sample_per_class = len(real_testloader.dataset) // real_testloader.dataset.class_num
    scores_real_inter_class = []
    scores_real_intra_class = []
    print(len(real_testloader.dataset), real_testloader.dataset.class_num)
    for i in range(0, len(real_testloader.dataset), sample_per_class):
        scores_real_inter_class.append(scores_real_all[i: i + sample_per_class, i + sample_per_class:].flatten())
        scores_real_intra_class.append(scores_real_all[i: i + sample_per_class, i: i + sample_per_class][np.triu_indices(sample_per_class, 1)].flatten())
    scores_real_inter_class = np.concatenate(scores_real_inter_class)
    scores_real_intra_class = np.concatenate(scores_real_intra_class)
    print(len(scores_real_intra_class))
    print(len(scores_real_inter_class))

    from matplotlib import pyplot as plt
    bin = 100
    plt.figure(1)
    plt.hist(scores_real_intra_class, bins=np.linspace(0, 1, bin), density=True, histtype='step', label='intra-class similarity (testset)')
    plt.hist(scores_real_inter_class, bins=np.linspace(0, 1, bin), density=True, histtype='step', label='inter-class similarity (testset)')
    plt.hist(scores_sty_sty, bins=np.linspace(0, 1, bin), density=True, histtype='step', label='inter-sample similarity (synthetic set)')
    plt.legend()
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.savefig("fv1.png")
    return 0


def save_roc():
    model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/fingerrec_gan-master/snapshots/diff_num/seed=99_dataset=FVUSM_network=resnet18_loss=contrast_BestROC=1.14_Epoch=73.pth"
    fv_unsupervised = compute_roc_from_snapshot(model_path)
    import scipy.io as sio
    sio.savemat('fv_roc_ablation.mat', {'fv_unsupervised': fv_unsupervised})


if __name__ == '__main__':
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/fingervein/supervised/seed=1_dataset=FVUSM_network=resnet18_loss=tripletCosface_1614945708_FinalROC=0.45_Epoch=79.pth"
    # analyze_pair_distance_new(model_path)
    model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/fingervein/supervised/seed=1_dataset=FVUSM_network=resnet18_loss=tripletCosface_1614945708_FinalROC=0.45_Epoch=79.pth"
    compute_roc_from_snapshot(model_path)
    # save_roc()

