#!/bin/bash
cd /remote-home/share/GraphMIL/backbone_aggregation/

re_dir=$2
aggregator=$3

CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "$aggregator" \
--load "/remote-home/share/GraphMIL/experiments/NCTCRC/MoCo/12.9+bs128+res50+mocoV2/checkpoint_0099.pth" \
--pretrained_type "moco" \
--log_dir "/remote-home/share/GraphMIL/experiments/BackboneAggregation/NCTCRC/moco/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
-cn 2 \
"/remote-home/share/DATA/NCTCRC/" &&
CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
-e \
--aggregator "$aggregator" \
--load "/remote-home/share/GraphMIL/experiments/NCTCRC/SimpleMIL/MTaugmentation/lr0.001_withval/checkpoint_0099.pth" \
--pretrained_type "oracle" \
--resume "/remote-home/share/GraphMIL/experiments/BackboneAggregation/NCTCRC/moco/$re_dir/$aggregator/model_best.pth" \
--log_dir "/remote-home/share/GraphMIL/experiments/BackboneAggregation/NCTCRC/moco/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
-cn 2 \
"/remote-home/share/DATA/NCTCRC/"
