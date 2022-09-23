#!/bin/bash
cd /remote-home/my/GraphMIL/backbone_aggregation/

## 2021.2.19 21:45 Oracle lr 0.001, max pooling
# CUDA_VISIBLE_DEVICES=1 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"