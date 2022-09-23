#!/bin/bash
cd /remote-home/my/GraphMIL/backbone_aggregation/
## std
CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "attention" \
--assumption "std" \
-e \
--load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
--resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_attention_pooling/model_best.pth' \
--pretrained_type "oracle" \
--log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/cross/train_count/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/" &&
## thres
CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \

--lr 0.001 \
--aggregator "attention" \
--assumption count --pos_ratio 0.2 \
-e \
--load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
--resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_attention_pooling/model_best.pth' \
--pretrained_type "oracle" \
--log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/cross/train_count/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/" &&
## co
CUDA_VISIBLE_DEVICES=0 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "attention" \
--assumption co --pos_targets 7 8 \
-e \
--load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
--resume '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CountMIL_0.2_0.5/lr0.001_attention_pooling/model_best.pth' \
--pretrained_type "oracle" \
--log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/cross/train_count/' \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"