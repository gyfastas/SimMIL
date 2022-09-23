#!/bin/bash
cd /remote-home/share/GraphMIL/backbone_aggregation/

re_dir=$2
aggr=$3
CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 -a resnet50 --lr 0.001 --assumption "std" --load "/remote-home/share/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.001_withval/model_best.pth" -cn 2 \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
--assumption "std" \
--pretrained_type "oracle" \
--aggregator $aggr \
--log_dir "/remote-home/share/GraphMIL/experiments/BackboneAggregation/NCTCRC/oracle/$re_dir/$aggr/" \
"/remote-home/share/DATA/NCTCRC/" &&
CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 -a resnet50 --lr 0.001 --assumption "std" --load "/remote-home/share/GraphMIL/experiments/NCTCRC/FSP/MTaugmentation/lr0.001_withval/model_best.pth" -cn 2 \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
--assumption "std" \
--pretrained_type "oracle" \
--aggregator $aggr \
-e \
--resume "/remote-home/share/GraphMIL/experiments/BackboneAggregation/NCTCRC/oracle/$re_dir/$aggr/model_best.pth" \
--log_dir "/remote-home/share/GraphMIL/experiments/BackboneAggregation/NCTCRC/oracle/$re_dir/$aggr/" \
"/remote-home/share/DATA/NCTCRC/" 
