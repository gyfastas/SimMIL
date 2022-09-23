#!/bin/bash
cd /remote-home/my/GraphMIL/backbone_aggregation/

## 2021.2.2 21:00 Debug: test mean pooling with oracle pretrained
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.1 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# -e \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.5 10:00 mean pooling with oracle pretrained, lr 0.1
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.1 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.1_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.6 11:40 mean pooling with simple MIL, lr 15.0
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"


## 2021.2.10 10:00 mean pooling with MoCoV1, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_mean_pooling/' \
# --resume "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_mean_pooling/model_best.pth" \
# -e \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.10 10:15 mean pooling with Oracle, lr 0.1 evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.1_mean_pooling/' \
# --resume "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.1_mean_pooling/model_best.pth" \
# -e \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.10 10:20 mean pooling with Oracle, lr 15.0 evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_mean_pooling/' \
# --resume "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_mean_pooling/model_best.pth" \
# -e \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.10 10:20 mean pooling with SimpleMIL, lr 15.0 evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_mean_pooling/' \
# --resume "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_mean_pooling/model_best.pth" \
# -e \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"


## 2021.2.12 9:20 max pooling with SimpleMIL
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "max_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.12 (?) Attention Pooling with SimpleMIL
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_attention/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:25 max pooling SimpleMIL evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "max_pooling" \
# --pretrained_type "oracle" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_max_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:30 attention pooling SimpleMIL evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "attention" \
# --pretrained_type "oracle" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_attention/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr15.0_attention/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:55 max pooling Moco V2, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "max_pooling" \
# --pretrained_type "moco" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_max_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:55 attention pooling Moco V2, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "attention" \
# --pretrained_type "moco" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_attention_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:55 attention pooling MT, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "attention" \
# --pretrained_type "oracle" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_attention_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:55 max pooling MT, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "max_pooling" \
# --pretrained_type "oracle" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_max_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:55 max pooling oracle, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "max_pooling" \
# --pretrained_type "oracle" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_max_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:55 attention pooling oracle, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "attention" \
# --pretrained_type "oracle" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_attention_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.14 23:55 attention pooling oracle lr 0.1
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.1 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "attention" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.1_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.15 8:25 SimpleMIL + lr 0.001 attention pooling
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_attention/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.16 16:40 SimpleMIL + lr 0.001 attention pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --pretrained_type "oracle" \
# -e \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_attention/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_attention/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.16 16:50 MoCoV1 + lr 0.001, max pooling
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.18 13:45 MoCoV1 + lr 15.0, GCN pooling
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "gcn" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr15.0_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 0:30 MoCoV1 + lr 0.001, GCN pooling
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 17:40 MoCoV1 + lr 0.001, GCN pooling evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --aggregator "gcn" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_gcn_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_gcn_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 18:00 MoCoV2 + lr0.001, GCN Pooling, evaluation
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# -e \
# --pretrained_type "moco" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_gcn_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_gcn_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 18:00 MT + lr 0.001, GCN Pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# -e \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_gcn_pooling/' \
# --resume "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_gcn_pooling/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 18:25 MoCoV2 + lr 0.001, attention pooling
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# -e \
# --pretrained_type "moco" \
# --aggregator "attention" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_attention_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_attention_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 18:40 Oracle lr0.001, gcn pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_gcn_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_gcn_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 21:40 Oracle lr0.001, attention pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "attention" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_attention_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_attention_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 22:00 MT lr 0.001, attention pooling, eval
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --pretrained_type "oracle" \
# --aggregator "attention" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_attention_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_attention_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 23:30 MoCoV1 + lr0.001, attention pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_attention_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_attention_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 23:40 MoCoV1 + lr0.001, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --aggregator "mean_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_mean_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_mean_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 23:40 MoCoV1 + lr0.001, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --aggregator "max_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_max_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_max_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 10:00 SimpleMIL + lr0.001, GCN pooling, evaluation
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_gcn_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_gcn_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 13:50 SimpleMIL + lr0.001, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_mean_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_mean_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 13:52 SimpleMIL + lr 0.001, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_max_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_max_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 23:54 MT + lr 0.001, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --pretrained_type "oracle" \
# --aggregator "mean_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_mean_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_mean_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:02 MT + lr 0.001, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_max_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_max_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:05 MoCoV2 + lr 0.001, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# -e \
# --pretrained_type "moco" \
# --aggregator "mean_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_mean_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_mean_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:20 MoCoV2 + lr 0.001, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# -e \
# --pretrained_type "moco" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_max_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_max_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:30 Oracle + lr 0.001, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_max_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_max_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:30 Oracle + lr 0.001, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "mean_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_mean_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_mean_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:40 Oracle 50% + lr0.001, gcn pooling
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 18:30 Oracle 50% + lr0.001, gcn pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_gcn_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_gcn_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 18:35 Oracle 50% + lr0.001, attention pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "attention" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_attention_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_attention_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 18:40 Oracle 50% + lr0.001, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "mean_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_mean_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_mean_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 18:40 Oracle 50% + lr0.001, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=4 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_max_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_max_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 19:15 Oracle ins max pooling evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/ins_max_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --pos_class 8 \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 19:55 SimpleMIL ins max pooling evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/ins_max_pooling/' \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --pos_class 1 \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 23:15 Oracle 50% + lr0.001 max pooling evaluation using assumption (debug)
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_max_pooling/' \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_max_pooling/model_best.pth' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 23:30 CountMIL, MT, 20% pos ratio
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.22 19:05 CountMIL, MT, 20% pos ratio, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_max_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.22 19:18 CountMIL, MT, 20% pos ratio, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_mean_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.22 19:18 CountMIL, MT, 20% pos ratio, attention pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_attention_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.22 19:18 CountMIL, MT, 20% pos ratio, gcn pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_gcn_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"


## 2021.2.22 19:30 CountMIL, MT, 50% pos ratio, max pooling
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_max_pooling/checkpoint_0034.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 21:05 CountMIL, MT, 50% pos ratio, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.5 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_max_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 21:30 Train on SimpleMIL, Test on CountMIL, 20% pos ratio, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_max_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 21:30 Train on SimpleMIL, Test on CountMIL, 20% pos ratio, attention pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_attention_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 22:20 Train on SimpleMIL, Test on CountMIL, 50% pos ratio, attention pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.5 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_attention_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 23:30 Train on SimpleMIL, Test on CountMIL, 50% pos ratio, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.5 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_max_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 23:55 Train on SimpleMIL, Test on CountMIL, 50% pos ratio, gcn pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# --pos_ratio 0.5 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_gcn_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.24 0:10 Train on CountMIL, 20% pos ratio, test on SimpleMIL, max pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "std" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_max_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/SimpleMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.24 0:10 Train on CountMIL, 20% pos ratio, test on SimpleMIL, attention pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "std" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_attention_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/SimpleMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.24 9:40 Train on CountMIL, 20% pos ratio, test on SimpleMIL, gcn pooling, evaluation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "std" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_gcn_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/SimpleMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

##  2021.2.24 23:20 CoMIL, MT, mean pooling
# CUDA_VISIBLE_DEVICES=2 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "co" \
# --pos_targets 7 8 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

##  2021.2.25 18:15 CoMIL, MT, max pooling
# CUDA_VISIBLE_DEVICES=3 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "co" \
# --pos_targets 7 8 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.26 14:05 CoMIL, MT, attention pooling
# CUDA_VISIBLE_DEVICES=3 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "co" \
# --pos_targets 7 8 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 15:00 CountMIL, 20%~50%, MT, attention
# CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.2 \
# --max_pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.28 11:50 Oracle + 50% aggregation redo
# CUDA_VISIBLE_DEVICES=1 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_gcn_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "mean_pooling" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_mean_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"