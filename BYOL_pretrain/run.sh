#!/bin/bash

dataset=FVUSM
#dataset=Palmvein
network=resnet18
max_epoch=80
batch_size=64
train_data="../vein_databases/synthetic_samples_fv"
#train_data="../vein_databases/synthetic_samples_pv"
#train_data="/home/weifeng/Desktop/Thesis_source_codes/chapter_5/vein_databases/synthetic_samples_pv"
#train_data="/home/weifeng/Desktop/Thesis_source_codes/chapter_5/vein_databases/synthetic_samples_fv"
#train_data="/home/weifeng/Desktop/Thesis_source_codes/chapter_5/vein_databases/synthetic_samples_pv"

for seed in 99; do
  for num in 500; do
    for momentum in 0.996; do
      timestamp=$(date +%s)
      python -u ./main.py \
        --seed $seed \
        --dataset $dataset --network $network \
        --batch_size ${batch_size} --max_epoch $max_epoch \
        --synthetic_num ${num} \
        --traindata ${train_data} \
        --encoder_momentum ${momentum} \
        2>&1 | tee trainlog_seed=${seed}_${dataset}_${network}_${loss}_${timestamp}_${num}.txt
    done
  done
done
