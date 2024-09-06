#!/bin/bash

seed=1
dataset=FVUSM
#dataset=Palmvein
data="../vein_databases/FV-USM-processed"
#data="../vein_databases/Palmvein_tongji"
#data="/home/weifeng/Desktop/Thesis_source_codes/chapter_5/vein_databases/FV-USM-processed"
network=resnet18
max_epoch=80
lr_decay_ep=60
for seed in 1; do
  # FusionAug
#  loss=tripletCosface
#  python -u ./train.py \
#      --seed $seed --max_epoch ${max_epoch} --lr_decay_ep ${lr_decay_ep} \
#      --dataset $dataset --data $data --network $network --loss ${loss} \
#      --pretrained \
#      --image_size 64 128 \
#      --inter_aug "TB" \
#      --intra_aug \
#      2>&1 | tee trainlog_seed=${seed}_${dataset}_${network}_${loss}_$(date +%s).txt
#
  # SimCLR+FusionAug
  snapshot="../SimCLR_pretrain/FinishModel/std_sfv_500/snapshots/seed=99_dataset=FVUSM_network=resnet18_loss=SimCLR_BestROC=3.90_Epoch=56.pth"
#  snapshot="/home/weifeng/Desktop/PycharmProjects/stylegan_paper/fingerrec_gan-master/snapshots/diff_num/sgd_mom=0.9_wd=5e-4_lr=0.01_two_session/seed=99_dataset=FVUSM_network=resnet18_loss=contrast_BestROC=1.16_Epoch=44.pth"
  loss=tripletCosface
  python -u ./train.py \
      --seed $seed --max_epoch ${max_epoch} --lr_decay_ep ${lr_decay_ep} \
      --dataset $dataset --data $data --network $network --loss ${loss} \
      --pretrained \
      --snapshot ${snapshot} \
      --image_size 64 128 \
      --inter_aug "TB" \
      --intra_aug \
      2>&1 | tee trainlog_seed=${seed}_${dataset}_${network}_${loss}_$(date +%s).txt

  # BYOL+FusionAug
  snapshot="../BYOL_pretrain/150_sfv_20000/snapshots/seed=99_dataset=FVUSM_network=resnet18_loss=BYOL_BestROC=6.42_Epoch=22.pth"
#  snapshot="/home/weifeng/Desktop/PycharmProjects/stylegan_paper/PyTorch-BYOL-master/snapshots/fv/seed=99_dataset=FVUSM_network=resnet18_loss=byol_BestROC=3.49_Epoch=30.pth"
  loss=tripletCosface
  python -u ./train.py \
      --seed $seed --max_epoch ${max_epoch} --lr_decay_ep ${lr_decay_ep} \
      --dataset $dataset --data $data --network $network --loss ${loss} \
      --pretrained \
      --snapshot ${snapshot} \
      --image_size 64 128 \
      --inter_aug "TB" \
      --intra_aug \
      2>&1 | tee trainlog_seed=${seed}_${dataset}_${network}_${loss}_$(date +%s).txt
done