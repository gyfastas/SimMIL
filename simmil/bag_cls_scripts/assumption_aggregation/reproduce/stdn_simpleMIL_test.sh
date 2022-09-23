#!/bin/bash
cd /remote-home/share/GraphMIL/backbone_aggregation/
re_dir=$2
aggregator=$3
CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "$aggregator" \
--assumption "std" \
--pos_targets 6 \
-e \
--resume "/remote-home/share/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL/std-N/$re_dir/$aggregator/model_best.pth" \
--pretrained_type "oracle" \
--log_dir "/remote-home/share/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL/std-N/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/NORM/train_ins_0.txt" \
--val_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/NORM/test_ins_0.txt" \
-cn 2 \
"/remote-home/share/DATA/NCTCRC/"
