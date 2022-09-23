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

## 2021.2.6 17:52 mean pooling with MoCoV1 pretrained, lr 15.0
# CUDA_VISIBLE_DEVICES=7 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV1/checkpoint_0100.pth" \
# --pretrained_type "moco" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MoCoV1/lr15.0_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.12 9:30 max pooling with mean teacher pretrained lr 15.0
# CUDA_VISIBLE_DEVICES=7 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "max_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.13 11:30 attention with mean teacher pretrained lr 15.0
# CUDA_VISIBLE_DEVICES=7 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr15.0_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"


## 2021.2.16 16:40 attention with mean teacher pretrained lr 0.001
# CUDA_VISIBLE_DEVICES=7 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.18 23:40 Oracle + lr 15.0, GCN pooling
# CUDA_VISIBLE_DEVICES=7 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 15.0 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr15.0_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 0:40 Oracle + lr 0.001, GCN pooling
# CUDA_VISIBLE_DEVICES=7 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.005_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "gcn" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.19 18:05 SimpleMIL + lr 0.001, max pooling
# CUDA_VISIBLE_DEVICES=7 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0050.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/SimpleMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.20 14:05 Oracle + 50%, lr 0.001, attention pooling
# CUDA_VISIBLE_DEVICES=7 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "attention" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.21 23:30 CountMIL, MT, 20% pos ratio, gcn
# CUDA_VISIBLE_DEVICES=3 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# --pos_ratio 0.2 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.22 19:50 CountMIL, MT, 50% pos ratio, gcn
# CUDA_VISIBLE_DEVICES=3 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# --pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_gcn_pooling/checkpoint_0032.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"


## 2021.2.22 19:50 CountMIL, MT, 50% pos ratio, gcn, evaluation
# CUDA_VISIBLE_DEVICES=2 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# --pos_ratio 0.5 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_gcn_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 15:00 CountMIL, 20%~50%, MT, gcn pooling
CUDA_VISIBLE_DEVICES=3 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "gcn" \
--assumption "count" \
--pos_ratio 0.2 \
--max_pos_ratio 0.5 \
--load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
--pretrained_type "oracle" \
--log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_gcn_pooling/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"
