#!/bin/bash
cd /remote-home/my/GraphMIL/backbone_aggregation/

re_dir=$2
aggregator=$3

CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "$aggregator" \
--assumption "std" \
--pos_targets 6 \
--load "/remote-home/my/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0099.pth" \
--pretrained_type "oracle" \
--log_dir "/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL/pos6/std/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NORM/train_ins_0.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NORM/val_ins_0.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/" &&
CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "$aggregator" \
--assumption "std" \
--pos_targets 6 \
-e \
--resume "/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL/pos6/std/$re_dir/$aggregator/model_best.pth" \
--pretrained_type "oracle" \
--log_dir "/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL/pos6/std/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NORM/train_ins_0.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NORM/test_ins_0.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"