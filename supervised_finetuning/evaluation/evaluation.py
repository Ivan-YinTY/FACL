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


def compute_roc_from_snapshot(model_path, is_byol=False):
    from networks.networks import ConvNet
    from networks import resnet
    from networks.vgg import vgg16_bn
    from torch.utils.data.dataloader import DataLoader

    snapshot = torch.load(model_path)
    args = snapshot['args']

    if args.dataset == 'FVUSM':
        sample_per_class = 12
        # data_path = "/home/weifeng/Desktop/FV-USM-processed"
        data_path = "../vein_databases/FV-USM-processed"
    elif args.dataset == "Palmvein" or args.dataset == "palmvein":
        sample_per_class = 20
        # data_path = "/home/weifeng/Desktop/Palmvein_tongji"
        data_path = "../vein_databases/Palmvein_tongji"
    else:
        raise ValueError('Dataset %s not exists!' % (args.dataset))

    from datasets.datasets import ImageDataset
    normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    testset = ImageDataset(root=data_path, sample_per_class=sample_per_class, transforms=transform_test, split='second_half', inter_aug="")
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if args.network == 'resnet18':
        model = resnet.resnet18(pretrained=True)
    elif args.network == 'resnet34':
        model = resnet.resnet34(pretrained=True)
    elif args.network == 'resnet50':
        model = resnet.resnet50(pretrained=True)
    elif args.network == 'vgg16':
        from networks.vgg import vgg16, vgg16_bn
        model = vgg16_bn(pretrained=True)
    else:
        raise ValueError('Network %s not supported!' % (args.network))

    import torchvision
    class ResNet18(torch.nn.Module):
        def __init__(self):
            super(ResNet18, self).__init__()
            resnet = torchvision.models.resnet18(pretrained=True)
            self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])

        def forward(self, x):
            h = torch.squeeze(self.encoder(x))
            return h

    if is_byol:
        model = ResNet18()

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in snapshot['model'].items() if k in model_dict and k != 'prelu_fc1.weight' and k != 'fc1.weight' and k != 'fc1.bias' and k != 'fc2.weight' and k != 'fc2.bias'}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda()

    fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets = compute_roc_two_session(model, testloader)
    # fpr, tpr, thresholds, scores_match, scores_imposter, embeddings, targets = compute_roc_two_session(model, testloader)
    roc_metrics, metrics_threds, auc = compute_roc_metrics(fpr, tpr, thresholds)
    print(metrics_threds)
    return fpr, tpr, thresholds, scores_match, scores_imposter


def save_roc():
    # 8000
    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/palmrec_gan-master/snapshots/diff_num/seed=99_dataset=palmvein_network=resnet18_loss=contrast_BestROC=3.62_Epoch=36.pth"
    # pv_nofil = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/palmrec_gan-master/snapshots/diff_num_filter_mean_std_0.005/seed=99_dataset=palmvein_network=resnet18_loss=contrast_1615540075_BestROC=3.51_Epoch=38.pth"
    # pv_fil_005 = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/palmrec_gan-master/snapshots/diff_num_filter_mean_std_0.006/seed=99_dataset=palmvein_network=resnet18_loss=contrast_1616165727_BestROC=3.27_Epoch=37.pth"
    # pv_fil_006 = compute_roc_from_snapshot(model_path)
    # 6000
    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/palmrec_gan-master/snapshots/diff_num/seed=99_dataset=palmvein_network=resnet18_loss=contrast_BestROC=4.25_Epoch=38.pth"
    # pv_nofil = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/palmrec_gan-master/snapshots/diff_num_filter_mean_std_0.005/seed=99_dataset=palmvein_network=resnet18_loss=contrast_1615539471_BestROC=3.45_Epoch=39.pth"
    # pv_fil_005 = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/palmrec_gan-master/snapshots/diff_num_filter_mean_std_0.006/seed=99_dataset=palmvein_network=resnet18_loss=contrast_1616165274_BestROC=3.11_Epoch=56.pth"
    # pv_fil_006 = compute_roc_from_snapshot(model_path)

    # import scipy.io as sio
    # plt.plot(pv_nofil[0], pv_nofil[1])
    # plt.plot(pv_fil_005[0], pv_fil_005[1])
    # plt.plot(pv_fil_006[0], pv_fil_006[1])
    # plt.show()
    # sio.savemat('palmvein_roc_8000.mat', {'pv_nofil': pv_nofil, 'pv_fil_005': pv_fil_005, "pv_fil_006": pv_fil_006})

    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/palmrec_gan-master/snapshots/diff_num_filter_mean_std_0.005/seed=99_dataset=palmvein_network=resnet18_loss=contrast_1615540762_BestROC=3.13_Epoch=33.pth"
    # pv_unsupervised = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/supervised/seed=1_dataset=Palmvein_network=resnet18_loss=softmax_BestROC=2.19_Epoch=38.pth"
    # pv_no_aug_softmax = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/supervised/seed=1_dataset=Palmvein_network=resnet18_loss=tripletCosface_1614943990_BestROC=1.30_Epoch=51.pth"
    # pv_aug_fusion = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/finetune/filter_mean_std_0.005/10000_finetune/seed=1_dataset=Palmvein_network=resnet18_loss=softmax_1615548424_BestROC=1.48_Epoch=43.pth"
    # pv_finetune_no_aug_softmax = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/finetune/filter_mean_std_0.005/10000_finetune/seed=1_dataset=Palmvein_network=resnet18_loss=tripletCosface_1615549127_BestROC=0.82_Epoch=48.pth"
    # pv_finetune_aug_fusion = compute_roc_from_snapshot(model_path)
    # import scipy.io as sio
    # sio.savemat('pv_roc_ablation.mat', {'pv_no_aug_softmax': pv_no_aug_softmax,
    #                                     "pv_aug_fusion": pv_aug_fusion,
    #                                     'pv_fin etune_no_aug_softmax': pv_finetune_no_aug_softmax,
    #                                     "pv_finetune_aug_fusion": pv_finetune_aug_fusion})

    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/fingerrec_gan-master/snapshots/diff_num/seed=99_dataset=FVUSM_network=resnet18_loss=contrast_BestROC=1.14_Epoch=73.pth"
    # fv_unsupervised = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/fingervein/supervised/seed=1_dataset=FVUSM_network=resnet18_loss=softmax_1614856157_BestROC=2.57_Epoch=75.pth"
    # fv_no_aug_softmax = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/fingervein/supervised/seed=1_dataset=FVUSM_network=resnet18_loss=tripletCosface_1614945495_BestROC=0.41_Epoch=25.pth"
    # fv_aug_fusion = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/fingervein/finetune/no_filter/10000_finetune/seed=1_dataset=FVUSM_network=resnet18_loss=softmax_1614856348_BestROC=0.66_Epoch=39.pth"
    # fv_finetune_no_aug_softmax = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/fingervein/finetune/no_filter/10000_finetune/seed=1_dataset=FVUSM_network=resnet18_loss=tripletCosface_1614856638_BestROC=0.21_Epoch=28.pth"
    # fv_finetune_aug_fusion = compute_roc_from_snapshot(model_path)
    # import scipy.io as sio
    # sio.savemat('fv_roc_ablation.mat', {'fv_no_aug_softmax': fv_no_aug_softmax,
    #                                     "fv_aug_fusion": fv_aug_fusion,
    #                                     'fv_finetune_no_aug_softmax': fv_finetune_no_aug_softmax,
    #                                     "fv_finetune_aug_fusion": fv_finetune_aug_fusion})

    # pv: TNNLS paper
    # import scipy.io as sio
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/supervised/seed=45_dataset=Palmvein_network=resnet18_loss=softmax_1614868191_BestROC=2.25_Epoch=52.pth"
    # pv_no_aug_softmax = compute_roc_from_snapshot(model_path)
    # sio.savemat("pv_no_aug_softmax,mat", {"pv_no_aug_softmax": pv_no_aug_softmax})
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/supervised/seed=1_dataset=Palmvein_network=resnet18_loss=tripletCosface_1614943990_BestROC=1.30_Epoch=51.pth"
    # pv_aug_fusion = compute_roc_from_snapshot(model_path)
    # sio.savemat("pv_aug_fusion,mat", {"pv_aug_fusion": pv_aug_fusion})

    # import scipy.io as sio
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/fingervein/supervised/seed=45_dataset=FVUSM_network=resnet18_loss=softmax_1614858183_BestROC=2.28_Epoch=79.pth"
    # fv_no_aug_softmax = compute_roc_from_snapshot(model_path)
    # sio.savemat("fv_no_aug_softmax,mat", {"fv_no_aug_softmax": fv_no_aug_softmax})
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/fingervein/supervised/seed=23_dataset=FVUSM_network=resnet18_loss=tripletCosface_1614945902_BestROC=0.35_Epoch=47.pth"
    # fv_aug_fusion = compute_roc_from_snapshot(model_path)
    # sio.savemat("fv_aug_fusion,mat", {"fv_aug_fusion": fv_aug_fusion})


    # new results: fv
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/fingervein/supervised/seed=1_dataset=FVUSM_network=resnet18_loss=softmax_BestROC=3.17_Epoch=75.pth"
    # fv_no_aug_softmax = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/fingervein/supervised/seed=1_dataset=FVUSM_network=resnet18_loss=tripletCosface_gamma_cos=1.0_gamma_tri=4.0_hard_margin=0.2_s=30.0_m=0.2_p=8_k=4_BestROC=0.57_Epoch=57.pth"
    # fv_aug_fusion = compute_roc_from_snapshot(model_path)
    #
    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/fingerrec_gan-master/snapshots/diff_num/sgd_mom=0.9_wd=5e-4_lr=0.01_two_session/seed=99_dataset=FVUSM_network=resnet18_loss=contrast_BestROC=1.16_Epoch=44.pth"
    # fv_simclr_20000 = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/fingervein/finetune/SimCLR_sgd_mom=0.9_wd=5e-4_lr=0.01_two_session/seed=1_dataset=FVUSM_network=resnet18_loss=softmax_1625277427_BestROC=0.53_Epoch=13.pth"
    # fv_simclr_no_aug_softmax = compute_roc_from_snapshot(model_path)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/fingervein/finetune/SimCLR_sgd_mom=0.9_wd=5e-4_lr=0.01_two_session/seed=1_dataset=FVUSM_network=resnet18_loss=tripletCosface_1625275755_BestROC=0.21_Epoch=38.pth"
    # fv_simclr_aug_fusion = compute_roc_from_snapshot(model_path)
    #
    # model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/PyTorch-BYOL-master/snapshots/fv/seed=99_dataset=FVUSM_network=resnet18_loss=byol_BestROC=3.49_Epoch=30.pth"
    # fv_byol_20000 = compute_roc_from_snapshot(model_path, True)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/fingervein/finetune/BYOL/20000/seed=1_dataset=FVUSM_network=resnet18_loss=softmax_1625234203_BestROC=1.04_Epoch=54.pth"
    # fv_byol_no_aug_softmax = compute_roc_from_snapshot(model_path, True)
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/fingervein/finetune/BYOL/20000/seed=1_dataset=FVUSM_network=resnet18_loss=tripletCosface_1625219838_BestROC=0.42_Epoch=33.pth"
    # fv_byol_aug_fusion = compute_roc_from_snapshot(model_path, True)
    #
    # import scipy.io as sio
    # sio.savemat('fv_roc_ablation.mat', {'fv_no_aug_softmax': fv_no_aug_softmax,
    #                                     "fv_aug_fusion": fv_aug_fusion,
    #                                     "fv_simclr_20000": fv_simclr_20000,
    #                                     'fv_simclr_no_aug_softmax': fv_simclr_no_aug_softmax,
    #                                     "fv_simclr_aug_fusion": fv_simclr_aug_fusion,
    #                                     "fv_byol_20000": fv_byol_20000,
    #                                     'fv_byol_no_aug_softmax': fv_byol_no_aug_softmax,
    #                                     "fv_byol_aug_fusion": fv_byol_aug_fusion,
    #                                     })

    # new results: pv
    model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/palmvein/supervised/seed=1_dataset=Palmvein_network=resnet18_loss=softmax_1625377518_BestROC=2.84_Epoch=72.pth"
    pv_no_aug_softmax = compute_roc_from_snapshot(model_path)
    model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/palmvein/supervised/seed=1_dataset=Palmvein_network=resnet18_loss=tripletCosface_1625379540_BestROC=1.66_Epoch=77.pth"
    pv_aug_fusion = compute_roc_from_snapshot(model_path)

    model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/palmrec_gan-master/snapshots/diff_num/80ep_sgd_lr=0.01_two_session/seed=99_dataset=palmvein_network=resnet18_loss=contrast_1625204821_BestROC=4.62_Epoch=52.pth"
    pv_simclr_10000 = compute_roc_from_snapshot(model_path)
    model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/palmvein/finetune/SimCLR_sgd_lr=0.01_two_session/10000/seed=1_dataset=Palmvein_network=resnet18_loss=softmax_1625311424_BestROC=1.92_Epoch=74.pth"
    pv_simclr_no_aug_softmax = compute_roc_from_snapshot(model_path)
    model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/palmvein/finetune/SimCLR_sgd_lr=0.01_two_session/10000/seed=1_dataset=Palmvein_network=resnet18_loss=tripletCosface_1625282007_BestROC=1.07_Epoch=37.pth"
    pv_simclr_aug_fusion = compute_roc_from_snapshot(model_path)

    model_path = "/home/weifeng/Desktop/PycharmProjects/stylegan_paper/PyTorch-BYOL-master/snapshots/pv/seed=99_dataset=Palmvein_network=resnet18_loss=byol_BestROC=7.77_Epoch=43.pth"
    pv_byol_15000 = compute_roc_from_snapshot(model_path, True)
    model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/palmvein/finetune/BYOL/15000/seed=1_dataset=Palmvein_network=resnet18_loss=softmax_1625293792_BestROC=2.90_Epoch=61.pth"
    pv_byol_no_aug_softmax = compute_roc_from_snapshot(model_path, True)
    model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/palmvein/finetune/BYOL/15000/seed=1_dataset=Palmvein_network=resnet18_loss=tripletCosface_1625294272_BestROC=2.04_Epoch=4.pth"
    pv_byol_aug_fusion = compute_roc_from_snapshot(model_path, True)

    import scipy.io as sio
    sio.savemat('pv_roc_ablation.mat', {'pv_no_aug_softmax': pv_no_aug_softmax,
                                        "pv_aug_fusion": pv_aug_fusion,
                                        "pv_simclr_10000": pv_simclr_10000,
                                        'pv_simclr_no_aug_softmax': pv_simclr_no_aug_softmax,
                                        "pv_simclr_aug_fusion": pv_simclr_aug_fusion,
                                        "pv_byol_15000": pv_byol_15000,
                                        'pv_byol_no_aug_softmax': pv_byol_no_aug_softmax,
                                        "pv_byol_aug_fusion": pv_byol_aug_fusion,
                                        })


if __name__ == '__main__':
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/seed=1_dataset=Palmvein_network=resnet18_loss=tripletCosface_1614943990_BestROC=1.30_Epoch=51.pth"
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/seed=1_dataset=FVUSM_network=resnet18_loss=tripletCosface_1615381657_FinalROC=0.85_Epoch=79.pth"
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/supervised/seed=45_dataset=Palmvein_network=resnet18_loss=softmax_1614868191_BestROC=2.25_Epoch=52.pth"
    # model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/palmvein/finetune/no_filter/10000_finetune/seed=45_dataset=Palmvein_network=resnet18_loss=softmax_1614868708_BestROC=1.42_Epoch=45.pth"
    model_path = "/home/weifeng/Downloads/FVR_release/FVR_release/snapshots/new_results_two_session/palmvein/finetune/BYOL/finetune_diff_num_one_session/seed=1_dataset=Palmvein_network=resnet18_loss=tripletCosface_1629461467_BestROC=1.16_Epoch=45.pth"
    compute_roc_from_snapshot(model_path, is_byol=True)
    # save_roc()


# 10:31:42 Performance evaluation...
# EER:0.48%, FRR@FAR=0.01: 0.22%, FRR@FAR=0.001: 1.58%, FRR@FAR=0.0001: 3.22%, FRR@FAR=0: 4.55%, Aver: 2.01%, AUC:99.99%
# 0.64685583

# 09:40:32 Performance evaluation...
# EER:0.97%, FRR@FAR=0.01: 0.95%, FRR@FAR=0.001: 3.20%, FRR@FAR=0.0001: 7.11%, FRR@FAR=0: 28.19%, Aver: 8.08%, AUC:99.76%
# (0.58667886, 0.5858886, 0.6387358, 0.67843735, 0.7728636)

# 09:41:44 Performance evaluation...
# EER:1.03%, FRR@FAR=0.01: 1.03%, FRR@FAR=0.001: 3.85%, FRR@FAR=0.0001: 8.87%, FRR@FAR=0: 33.31%, Aver: 9.62%, AUC:99.76%
# (0.5797696, 0.58044046, 0.6340784, 0.6794783, 0.77989703)

# 09:42:32 Performance evaluation...
# EER:1.15%, FRR@FAR=0.01: 1.24%, FRR@FAR=0.001: 3.82%, FRR@FAR=0.0001: 8.53%, FRR@FAR=0: 23.77%, Aver: 7.70%, AUC:99.74%
# (0.58664393, 0.5903504, 0.6435883, 0.68521947, 0.7569649)

# 09:43:32 Performance evaluation...
# EER:1.11%, FRR@FAR=0.01: 1.21%, FRR@FAR=0.001: 3.72%, FRR@FAR=0.0001: 7.61%, FRR@FAR=0: 23.87%, Aver: 7.50%, AUC:99.75%
# (0.5757942, 0.5785812, 0.63017094, 0.6711552, 0.752188)

# 09:44:29 Performance evaluation...
# EER:1.02%, FRR@FAR=0.01: 1.03%, FRR@FAR=0.001: 3.14%, FRR@FAR=0.0001: 6.74%, FRR@FAR=0: 21.09%, Aver: 6.60%, AUC:99.76%
# (0.58226013, 0.5828281, 0.63364434, 0.6736611, 0.7441485)

# EER:1.16%, FRR@FAR=0.01: 1.24%, FRR@FAR=0.001: 4.12%, FRR@FAR=0.0001: 8.55%, FRR@FAR=0: 24.51%, Aver: 7.92%, AUC:99.76%
# (0.5798424, 0.58384585, 0.6377939, 0.68034184, 0.75490844)