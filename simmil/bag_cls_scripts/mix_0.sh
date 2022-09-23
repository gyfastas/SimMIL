#!/bin/bash
cd /remote-home/my/GraphMIL/backbone_aggregation/

## 2021.2.19 18:15 MT lr 0.001, mean-pooling
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --aggregator "mean_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/MT/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.26 16:55 ImgNet Pretrained, lr 0.001, mean pooling
# CUDA_VISIBLE_DEVICES=6 python main.py --runner base -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --imgnet_pretrained \
# --pretrained_type "oracle" \
# --aggregator "mean_pooling" \
# --log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/ImgNetPT/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" && touch mix_0.finish

# Train on CountMIL, 20% pos ratio, test on SimpleMIL, gcn pooling, evaluation
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
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


## 2021.2.27 15:30 Train on CountMIL 0.2, test on CountMIL 0.5 
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.5 \
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

# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.5 \
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

# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.5 \
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

# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# --pos_ratio 0.5 \
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


## 2021.2.27 18:15 (?) Train on Threshold based [50%], test on stdMIL 
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "std" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_gcn_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/SimpleMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "std" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_attention_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/SimpleMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "std" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_max_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/SimpleMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "std" \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_mean_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/SimpleMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 19:30 (?) Train on Threshold based [50%], test on Threshold based [20%]
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_gcn_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_attention_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_max_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# --pos_ratio 0.2 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.5/lr0.001_mean_pooling/model_best.pth' \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.27 23:30 CoMIL Testing
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "co" \
# --pos_targets 7 8 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_gcn_pooling/model_best.pth'  \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "co" \
# --pos_targets 7 8 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_attention_pooling/model_best.pth'  \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "co" \
# --pos_targets 7 8 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_max_pooling/model_best.pth'  \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "co" \
# --pos_targets 7 8 \
# -e \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_mean_pooling/model_best.pth'  \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.28 1:30 CountMIL 0.2 0.5 Testing
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "count" \
# -e \
# --pos_ratio 0.2 \
# --max_pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_attention_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "gcn" \
# --assumption "count" \
# -e \
# --pos_ratio 0.2 \
# --max_pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_gcn_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_gcn_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "max_pooling" \
# --assumption "count" \
# -e \
# --pos_ratio 0.2 \
# --max_pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_max_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_max_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/" &&
# CUDA_VISIBLE_DEVICES=5 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "mean_pooling" \
# --assumption "count" \
# -e \
# --pos_ratio 0.2 \
# --max_pos_ratio 0.5 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_mean_pooling/model_best.pth' \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_mean_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/test" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

## 2021.2.28 11:50 Oracle + 50% aggregation redo
CUDA_VISIBLE_DEVICES=5 python main.py --runner base -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--load "/remote-home/my/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation_0.5dataset/lr0.001_withval/model_best.pth" \
--pretrained_type "oracle" \
--aggregator "mean_pooling" \
--log_dir '/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/re1/lr0.001_mean_pooling/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"