# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=3
#for i in 1 2 3
#for i in 1
#do

#python -u train.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 20  -alpha 0.1 -wd 0.0001  -imgset train
#python -u vis_fig2.py -net tg -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 100   -alpha 0.1 -wd 0.0001  -imgset train
#python -u vis_fig2.py -net tg -b 4 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 100   -alpha 0.1 -wd 0.0001  -imgset train
#python -u vis_fig2.py -net tg -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 50   -alpha 0.1 -wd 0.0001  -imgset all -iter 2500
#python -u vis_fig2.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 999   -alpha 0.1 -wd 0.0001  -imgset all -iter 40000
#python -u train1.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000  -alpha 0.1 -wd 0.0001  -imgset train -iter 40000
#python -u train1.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000  -alpha 0.1 -wd 0.0001  -imgset train -iter 40000
#python -u train.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000  -alpha 0.1 -wd 0.0001  -imgset train -iter 40000
#python -u vis_fig2.py -net tg -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 50   -alpha 0.1 -wd 0.0001  -imgset all -iter 2500
#python -u vis_fig2.py -net tg -b 4 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 50   -alpha 0.1 -wd 0.0001  -imgset all -iter 2500
#python -u train_m.py -net tgt -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000   -alpha 0.1 -wd 0.0001
#python -u train.py -net tg -b 16 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 2000   -alpha 0.1 -wd 0.0005
#python -u train.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 200   -alpha 0.1 -imgset train
#python -u train.py -net tgt -b 8 -lr 0.004 -min_lr 0.0004  -dataset crag -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 200   -alpha 0.1 -imgset train
# export PATH="/home/baiyu/miniconda3/envs/torch1.13/bin:$PATH"
eval "$(/home/baiyu/miniconda3/condabin/conda shell.bash hook)"
which conda
conda activate torch1.13
# import sys
# sys.path.insert(0, '/home/baiyu/miniconda3/envs/torch1.13/lib/python3.10/site-packages')

# import numpy
# print(numpy.__file__)
# source xxn
# # ls "/home/baiyu/miniconda3/etc/profile.d/conda.sh"
# which conda
# conda env list 
# # conda activate "torch1.13"
# /home/baiyu/miniconda3/condabin/conda activate "torch1.13"
# 获取开始时间（精确到秒），使用date命令获取当前时间并格式化输出
start_time=$(date +%s)
# python -u train.py -net tgt  -iter 2000  -rate 30 -b 8 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 100   -alpha 0.1 -imgset train
# python -u train.py -net tgt  -iter 2000  -rate 35 -b 8 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 100   -alpha 0.1 -imgset train
# python -u train.py -net tgt  -iter 2000  -rate 40 -b 8 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 100   -alpha 0.1 -imgset train
python -u train.py -net tgt  -iter 2000  -rate 25 -b 8 -lr 0.004 -min_lr 0.0004  -dataset Glas -gpu -download  -prefix unet_branch_SGD_473 -poly  -fp16 -eval_iter 100   -alpha 0.1 -imgset train

# 获取结束时间（精确到秒）
end_time=$(date +%s)
# 计算时间差（单位：秒）
time_diff=$((end_time - start_time))

# 将开始时间、结束时间、时间差信息写入time.txt文件
# echo "开始时间: $(date -d @$start_time +'%Y-%m-%d %H:%M:%S')" > time.txt
# echo "结束时间: $(date -d @$end_time +'%Y-%m-%d %H:%M:%S')" >> time.txt
# echo "时间差（秒）: $time_diff" >> time1041.txt
# echo "时间差（秒）: $time_diff" >> time1042.txt
# echo "时间差（秒）: $time_diff" >> time1043.txt
# echo "时间差（秒）: $time_diff" >> time1044.txt
#done
