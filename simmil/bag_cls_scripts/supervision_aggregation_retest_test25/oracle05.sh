#!/bin/bash
cd /remote-home/my/GraphMIL/backbone_aggregation/
re_dir=$2
aggregator=$3
CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
-e \
--aggregator "$aggregator" \
--pretrained_type "oracle" \
--resume "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/$re_dir/$aggregator/model_best.pth" \
--log_dir "/remote-home/my/GraphMIL/experiments/BackboneAggregation/NCTCRC/Oracle_0.5/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
-cn 2 \
"/remote-home/source/DATA/NCTCRC/"