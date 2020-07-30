#python main_FA.py --lr 0.0001 --cos --epochs 100  --model attention --pretrained None
#python main_FA.py --lr 0.001 --cos --epochs 100  --model gated_attention --pretrained None
#python main_FA.py --lr 0.0001 --cos --epochs 100  --model gated_attention --pretrained None
#python main_FA.py --lr 0.001 --cos --epochs 100  --model gated_attention
#python main_FA.py --lr 0.01 --cos --epochs 100  --model gated_attention
#python main_FA.py --lr 1e-5 --cos --epochs 200  --model gated_attention --pretrained None --weight 10
#python main_FA.py --lr 1e-5 --cos --epochs 200  --model gated_attention --pretrained None --weight 2
#python main_FA.py --lr 1e-5 --cos --epochs 200  --model gated_attention --pretrained None --weight 0.1
#python main_FA.py --lr 1e-5 --cos --epochs 200  --model attention --pretrained None --weight 10
#python main_FA.py --lr 1e-5 --cos --epochs 200  --model attention --pretrained None --weight 2
#python main_FA.py --lr 1e-5 --cos --epochs 200  --model attention --pretrained None --weight 0.1
# epochs=200, reg 1e-4, seed=1, cuda=True, model=gated,mil=0(attentionMIL),
#insnorm=False(Using BN), not using schedule( using when not cos), a=resnet18
#data_dir, log_dir, config
#python main_FA.py --data_dir '/remote-home/my/datasets/BASH' --cos --pretrained  \
#                  --lr 1e-5 --weight
python main_FA.py --model gated_attention --lr 1e-5 --cos --pretrained  --weight 0 --folder 0
python main_FA.py --model gated_attention --lr 1e-5 --cos --pretrained  --weight 0 --folder 1
python main_FA.py --model gated_attention --lr 1e-5 --cos --pretrained  --weight 0 --folder 2
python main_FA.py --model gated_attention --lr 1e-5 --cos --pretrained  --weight 0 --folder 3
python main_FA.py --model gated_attention --lr 1e-5 --cos --pretrained  --weight 0 --folder 4