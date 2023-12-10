#echo '============bail============='
#CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset bail --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1
#
#echo '============bail============='
#CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset bail --seed_num 10 --tem 0.5 --gamma 0.1 --lr_w 1
#
#echo '============pokec_z============='
#CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset pokec_z --seed_num 10 --tem 0.5 --gamma 0.001 --lr_w 1

echo '============pokec_z============='
CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset pokec_z --seed_num 10 --tem 0.9 --gamma 0.001 --lr_w 1

#echo '============pokec_n============='
#CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset pokec_n --seed_num 10 --tem 0.9 --gamma 0.001 --lr_w 1
#
#echo '============pokec_n============='
#CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gin --dataset pokec_n --seed_num 10 --tem 0.9 --gamma 0.001 --lr_w 1
