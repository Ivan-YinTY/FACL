#!/bin/bash

seed=99
dataset=FVUSM
#dataset=Palmvein
network=resnet18
max_epoch=80
batch_size=256
#train_data="../GSCL_Res/FV_Samp150_p"
#train_data="../vein_databases/synthetic_samples_fv"
train_data="../vein_databases/ADA-seg"
for num in 500; do
  timestamp=$(date +%s)
  python -u ./train.py \
    --seed $seed \
    --dataset $dataset --network $network \
    --batch_size ${batch_size} --max_epoch $max_epoch \
    --pretrained \
    --synthetic_num ${num} \
    --traindata ${train_data} \
    2>&1 | tee trainlog_seed=${seed}_${dataset}_${network}_${timestamp}_${num}.txt
done