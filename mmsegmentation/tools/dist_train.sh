#CONFIG=$1
#GPUS=$2
#NNODES=${NNODES:-1}
#NODE_RANK=${NODE_RANK:-0}
#PORT=${PORT:-29500}
#MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch \
#    --nnodes=$NNODES \
#    --node_rank=$NODE_RANK \
#    --master_addr=$MASTER_ADDR \
#    --nproc_per_node=$GPUS \
#    --master_port=$PORT \
#    $(dirname "$0")/train.py \
#    $CONFIG \
#    --launcher pytorch ${@:3}



export CUDA_VISIBLE_DEVICES=0
python tools/train.py /data/hdd1/by/House-Prices-Advanced-Regression-Techniques/mmsegmentation/configs/unet/fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_glas.py
