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

## 2021.2.5 10:00 mean pooling with oracle pretrained lr 15.0
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.6 11:45 mean pooling with mean teacher pretrained lr 15.0
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.10 10:00 mean pooling with mean teacher pretrained lr 15.0
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_mean_pooling/' \
# --resume "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_mean_pooling/model_best.pth" \
# -e \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.12 9:30 max pooling oracle, lr 15.0
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "max_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.12 9:30 attention pooling oracle, lr 15.0
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.15 attention pooling oracle, lr 0.001
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.16 16:50 MoCoV1 + lr 0.001, attention pooling
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.18 14:10 MoCoV2 + lr15.0, GCN Pooling
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr15.0_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 0:30 MoCoV2 + lr0.001, GCN Pooling
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV2/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 18:00 SimpleMIL + lr 0.001, GCN Pooling
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:05 Oracle + 50%, lr 0.001, mean pooling
# CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 23:15 Oracle 50% + lr0.001 max pooling evaluation using count assumption (debug)
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --assumption "count" \
# --pos_ratio 0.2 \
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

## 2021.2.21 23:30 CountMIL, MT, 20% pos ratio, attention 
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.2 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.22 19:45 CountMIL, MT, 50% pos ratio, attention 
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_attention_pooling/checkpoint_0034.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"


## 2021.2.23 21:05 CountMIL, MT, 50% pos ratio, attention, evaluation
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.5 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_attention_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 21:30 Train on SimpleMIL, Test on CountMIL, 20% pos ratio, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_mean_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 21:30 Train on SimpleMIL, Test on CountMIL, 20% pos ratio, gcn pooling, evaluation
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_gcn_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.23 22:20 Train on SimpleMIL, Test on CountMIL, 50% pos ratio, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.5 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_mean_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.24 0:10 Train on CountMIL, 20% pos ratio, test on SimpleMIL, mean pooling, evaluation
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "std" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_mean_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/SimpleMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.24 0:10 Train on CountMIL, 20% pos ratio, test on SimpleMIL, GCN pooling, evaluation
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
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

## 2021.2.26 14:05 CoMIL, MT, gcn pooling
# CUDA_VISIBLE_DEVICES=4 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "co" \
# --pos_targets 7 8 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 15:00 CountMIL, 20%~50%, MT, max pooling
# CUDA_VISIBLE_DEVICES=1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# --max_pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.28 11:50 Oracle + 50% aggregation redo
# CUDA_VISIBLE_DEVICES=2 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "attention" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"
CUDA_VISIBLE_DEVICES=2 python main.py --runner base -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
-e \
--load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
--pretrained_type "oracle" \
--aggregator "attention" \
--resume '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_attention_pooling/model_best.pth' \
--log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_attention_pooling/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"