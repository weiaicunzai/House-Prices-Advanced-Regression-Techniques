export CUDA_VISIBLE_DEVICES=2
#for i in 1 2 3
#for i in 1
#do

#python -u train.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000   -alpha 0.1 -wd 0.0001 
python -u train.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000   -alpha 0.1 -wd 0.0005 
#python -u train.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000   -alpha 0.1
#done
