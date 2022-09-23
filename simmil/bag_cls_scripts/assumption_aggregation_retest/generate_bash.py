import argparse
import sys

import os


if __name__=="__main__":
    pooling_methods = ["max_pooling", "mean_pooling", "attention", "gcn"]
    pooling_dirs = ["max_pooling", "mean_pooling", "attention_pooling", "gcn_pooling"]


    exp_folders = ["CoMIL", "CountMIL", "CountMIL_0.2_0.5", "CountMIL_0.5", "SimpleMIL"]
    assumptions = ["co", "count", "count", "count", "std"]
    extra_patterns = ["--pos_targets 7 8", "--pos_ratio 0.2", "--pos_ratio 0.2 --max_pos_ratio 0.5", "--pos_ratio 0.5", ""]
    exp_prefix = "lr0.001"
    gpu = 6
    exp_root = "/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/"

    with open("run.sh", "a+") as f:
        for (exp_folder, assumption, extra_pattern) in zip(exp_folders, assumptions, extra_patterns):
            for (pool_method, pooling_dir) in zip(pooling_methods, pooling_dirs):
                fixed_pattern = "CUDA_VISIBLE_DEVICES={} python main.py --runner assumption -b 4 -j 16 --gpu 0 -e --pretrained_type oracle -cn 2".format(gpu)
                sub_dirs = "--train-subdir NCT-CRC-HE-100K --eval-subdir VAL_TEST/test"
                label_files = "--train_label_file /remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt --val_label_file /remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/test_ins.txt"
                resume_patterns = "--resume {}".format(os.path.join(exp_root, exp_folder, exp_prefix+"_"+pooling_dir, "model_best.pth"))
                agg_patterns = "--aggregator {}".format(pool_method)
                assumption_pattern = "--assumption {}".format(assumption)
                log_dir_pattern = "--log_dir {}".format(os.path.join(exp_root, exp_folder, exp_prefix+"_"+pooling_dir,))
                end_pattern = "/remote-home/source/DATA/NCTCRC/ &&\n"


                cmd = """{} {} {} {} {} {} {} {} {}""".format(fixed_pattern, sub_dirs, label_files, resume_patterns, agg_patterns, assumption_pattern, extra_pattern, log_dir_pattern, end_pattern)

                f.write(cmd)
        


# CUDA_VISIBLE_DEVICES=3 python main.py --runner assumption -b 4 -j 16 --gpu 0 \
# -a resnet50 \
# --lr 0.001 \
# --aggregator "attention" \
# --assumption "co" \
# --pos_targets 7 8 \
# --load "/remote-home/my/GraphMIL/experiments/NCTCRC/MeanTeacher/Pretrained_200epochs/model_best.pth" \
# --pretrained_type "oracle" \
# --log_dir '/remote-home/my/GraphMIL/experiments/AssumptionAggregation/NCTCRC/MT/CoMIL/lr0.001_attention_pooling/' \
# --train-subdir "NCT-CRC-HE-100K" \
# --eval-subdir "VAL_TEST/val" \
# --train_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
# --val_label_file "/remote-home/my/GraphMIL/samples/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
# -cn 2 \
# "/remote-home/source/DATA/NCTCRC/"

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


