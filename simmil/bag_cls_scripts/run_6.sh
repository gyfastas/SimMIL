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

## 2021.2.5 10:00 mean pooling with oracle pretrained, lr 0.01
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.01 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.01_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.6 17:52 mean pooling with MoCoV2 pretrained, lr 15.0
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.10 10:10 mean pooling with MoCoV2 pretrained, evaluation
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_mean_pooling/' \
# --resume "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_mean_pooling/model_best.pth" \
# -e \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.12 9:25 max pooling with MoCoV2 pretrained, lr 15.0
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "max_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.13 11:30 attention pooling with MoCoV2 pretrained, lr 15.0
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.13 11:30 attention pooling with MoCoV2 pretrained, lr 0.001
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.16 16:50 MoCoV1 + lr 0.001, mean pooling
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/sasmples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.18 23:40 MT + lr15.0, gcn pooling
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.18 23:40 MT + lr0.001, gcn pooling
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 18:00 SimpleMIL + lr 0.001, mean pooling
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:05 Oracle + 50%, lr 0.001, mean pooling
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 23:30 CountMIL, MT, 20% pos ratio, mean pooling
# CUDA_VISIBLE_DEVICES=2 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.22 19:45 CountMIL, MT, 50% pos ratio, mean pooling
# CUDA_VISIBLE_DEVICES=2 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume "/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_mean_pooling/checkpoint_0033.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 21:10 CountMIL, MT, 50% pos ratio, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=2 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.5 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume "/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_mean_pooling/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"



## 2021.2.26 16:50 InsMIL max pooling, MT
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/ins_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.26 20:45 InsMIL prob max pooling, MT
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "prob_max" \
# --pos_class 8 \
# -e \
# --pretrained_type "mt" \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/prob_max/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 00:20 InsMIL prob mean pooling, MT
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "prob_mean" \
# --pretrained_type "mt" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/prob_mean/' \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 00:40 InsMIL prob max pooling, SimpleMIL
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "prob_max" \
# --pos_class 1 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/prob_max/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 00:40 InsMIL prob mean pooling, SimpleMIL
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "prob_mean" \
# --pos_class 1 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/prob_mean/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"


## 2021.2.27 00:40 InsMIL major vote, SimpleMIL
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "major_vote" \
# --pos_class 1 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/major_vote/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 10:20 major vote for mt
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "major_vote" \
# --pretrained_type "mt" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/major_vote/' \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 10:40 insMIL for Oracle
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "prob_max" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/prob_max/' \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "prob_mean" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/prob_mean/' \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "major_vote" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/major_vote/' \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 10:40 insMIL for Oracle 50%
# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "major_vote" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/major_vote/' \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "prob_max" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/prob_max/' \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner ins -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --ins_aggregation "prob_mean" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/prob_mean/' \
# --pos_class 8 \
# -e \
# --resume "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 9 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 12:05 ImgNet Pretrain testing
# CUDA_VISIBLE_DEVICES=2 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --pretrained_type "oracle" \
# --aggregator "mean_pooling" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_mean_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# -e \
# --lr 0.001 \
# --pretrained_type "oracle" \
# --aggregator "attention" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_attention_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# -e \
# --lr 0.001 \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_gcn_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=2 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# -e \
# --lr 0.001 \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_max_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

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

## 2021.2.27 15:00 CountMIL, 20%~50%, MT, mean pooling
# CUDA_VISIBLE_DEVICES=2 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# --max_pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.28 11:50 Oracle + 50% aggregation redo
# CUDA_VISIBLE_DEVICES=3 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.28 22:10 Oracle + 50% aggregation redo, testing
# CUDA_VISIBLE_DEVICES=3 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "max_pooling" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_max_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

CUDA_VISIBLE_DEVICES=3 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
-e \
--load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
--pretrained_type "oracle" \
--aggregator "mean_pooling" \
--resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_mean_pooling/model_best.pth' \
--log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_mean_pooling/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"