#!/bin/bash

#SBATCH --job-name=simpleMIL_camelyon16
#SBATCH --partition=dgx2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err

# export NCCL_DEBUG=INFO

# ulimit -s unlimited
# ulimit -l unlimited

# source /usr/share/Modules/init/bash

module purge
module load cuda/10.1.243-gcc-4.8.5
module load cudnn/7.6.5.32-10.1-linux-x64-gcc-4.8.5
module load miniconda3/4.6.14-gcc-4.8.5
module load gcc/5.5.0-gcc-4.8.5
source activate gyf_1.7

export PATH="/lustre/home/acct-seexy/seexy/.conda/envs/gyf_1.7/bin:$PATH"
cd /lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/backbone_aggregation/

re_dir=$2
aggregator=$3
python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "$aggregator" \
--assumption "std" \
--load '/lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/experiments/SimpleMIL/CameLyon16_lincls_lr0.001_withval_ckp99/' \
--pretrained_type "oracle" \
--log_dir "/lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL/std/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/val" \
--train_label_file "/lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/nctcrc_bags/BL50/VAL_TEST/val_ins.txt" \
-cn 2 \
'/lustre/home/acct-seexy/seexy/main/datasets/NCTCRC/' &&
python main.py --runner assumption -b 4 -j 16 --gpu 0 \
-a resnet50 \
--lr 0.001 \
--aggregator "$aggregator" \
--assumption "std" \
-e \
--resume  "/lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL/std/$re_dir/$aggregator/model_best.pth" \
--pretrained_type "oracle" \
--log_dir  "/lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/experiments/AssumptionAggregation/NCTCRC/SimpleMIL/std/$re_dir/$aggregator/" \
--train-subdir "NCT-CRC-HE-100K" \
--eval-subdir "VAL_TEST/test" \
--train_label_file "/lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/nctcrc_bags/BL50/NCT-CRC-HE-100K_ins.txt" \
--val_label_file "/lustre/home/acct-seexy/seexy/main/gyf/GraphMIL/nctcrc_bags/BL50/VAL_TEST/test_ins.txt" \
-cn 2 \
'/lustre/home/acct-seexy/seexy/main/datasets/NCTCRC/' 