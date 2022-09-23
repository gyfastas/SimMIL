#!/bin/bash
cd /remote-home/share/GraphMIL/backbone_aggregation/

re_dir=$2
aggregator=$3
target=$4
d_target=$5
CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "$aggregator" \
--assumption "std" \
--pos_targets $d_target \
--load "/remote-home/share/GraphMIL/experiments/BackboneCheck/NCTCRC_pretrain/simpleMIL-$target/1/lr0.001_withval/checkpoint_0099.pth" \
--pretrained_type "oracle" \
--log_dir "/remote-home/share/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL-$target/assumption-$d_target/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/target$d_target/train_ins_0.txt" \
--val_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/target$d_target/val_ins_0.txt" \
-cn 2 \
"/remote-home/share/DATA/NCTCRC/" &&
CUDA_VISIBLE_DEVICES=$1 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "$aggregator" \
--assumption "std" \
--pos_targets $d_target \
-e \
--resume "/remote-home/share/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL-$target/assumption-$d_target/$re_dir/$aggregator/model_best.pth" \
--pretrained_type "oracle" \
--log_dir "/remote-home/share/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL-$target/assumption-$d_target/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/target$d_target/train_ins_0.txt" \
--val_label_file "/remote-home/share/GraphMIL/samples/nctcrc_bags/BL50/target$d_target/test_ins_0.txt" \
-cn 2 \
"/remote-home/share/DATA/NCTCRC/"
