export CUDA_VISIBLE_DEVICES=1
#for i in 1 2 3
#for i in 1
#do

#python -u train.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 20  -alpha 0.1 -wd 0.0001  -imgset train
#python -u vis_fig2.py -net tg -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 100   -alpha 0.1 -wd 0.0001  -imgset train 
#python -u vis_fig2.py -net tg -b 4 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 100   -alpha 0.1 -wd 0.0001  -imgset train 
#python -u vis_fig2.py -net tg -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 50   -alpha 0.1 -wd 0.0001  -imgset all -iter 2500
#python -u vis_fig2.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 999   -alpha 0.1 -wd 0.0001  -imgset all -iter 40000
#python -u train1.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000  -alpha 0.1 -wd 0.0001  -imgset train -iter 40000
python -u train1.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000  -alpha 0.1 -wd 0.0001  -imgset train -iter 40000
#python -u vis_fig2.py -net tg -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 50   -alpha 0.1 -wd 0.0001  -imgset all -iter 2500
#python -u vis_fig2.py -net tg -b 4 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 50   -alpha 0.1 -wd 0.0001  -imgset all -iter 2500
#python -u train_m.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000   -alpha 0.1 -wd 0.0001  
#python -u train.py -net tg -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000   -alpha 0.1 -wd 0.0005 
#python -u train.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 200   -alpha 0.1 -imgset train 
#python -u train.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 200   -alpha 0.1 -imgset train 
#done
