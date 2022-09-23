#!/bin/bash
cd /remote-home/my/GraphMIL/backbone_aggregation/

## 2021.2.19 18:20 MoCoV2 lr0.001, max-pooling
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.26 16:55 ImgNet Pretrained, lr 0.001, attention pooling
CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--imgnet_pretrained \
--pretrained_type "oracle" \
--aggregator "attention" \
--log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_attention_pooling/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"