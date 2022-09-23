#!/bin/bash
cd /remote-home/my/GraphMIL/backbone_aggregation/

## 2021.2.19 21:45 Oracle lr 0.001, mean pooling
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "mean_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"


## 2021.2.26 17:40 ImgNet Pretrained, lr 0.001, gcn pooling
CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--imgnet_pretrained \
--pretrained_type "oracle" \
--aggregator "gcn" \
--log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_gcn_pooling/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"