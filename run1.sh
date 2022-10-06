export CUDA_VISIBLE_DEVICES=1
#for i in 1 2 3
for i in 1 
do
#python -u train.py -net segnet -b 6 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op xor -prefix baselinee1500lr0.01 -baseline
#python -u train.py -net segnet -b 6 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op xor -prefix e1500lr0.01a5opxor
#python -u train.py -net segnet -b 6 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 2 -op xor -prefix e1500lr0.01a2opxor
#python -u train.py -net segnet -b 6 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix e1500lr0.01a5opnone
#python -u train.py -net segnet -b 16 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix del
#python -u train.py -net hybird -b 16 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix del 
#python -u train.py -net hybird -b 16 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix del -fp16
python -u train.py -net fullnet -b 4 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix del -fp16
#python -u train.py -net transunet -b 16 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix del
#python -u train.py -net axialunet -b 4 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix axialcnn_SGD_473 -poly
#python -u train.py -net hybird -b 4 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix hybird_no_attn_branch_SGD_473 -poly
##python -u pretrain.py -net  transseg -b 4 -lr 0.001 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix transseg_SGD_473_poly_backbone_resnet50_fine_tune -pretrain
#python -u train.py -net  transseg -b 4 -lr 0.001 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix transseg_SGD_473_adam_lr_0.001_poly_decay_backbone_resnet50_c3  -poly 
#python -u train.py -net  transseg -b 4 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix transseg_SGD_473_adam_lr_0.01_poly_decay_backbone_resnet50_c3  -poly
#python -u train.py -net  transseg -b 4 -lr 0.001 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix transseg_SGD_473_adam_lr0.001_no_decay_backbone_resnet50_c3
#python -u train.py -net hybird -b 4 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix trans_branch_SGD_473 -poly -branch trans
# python -u train.py -net hybird -b 16 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix cnn_branch_SGD_473 -poly -branch cnn
#python -u train.py -net cnn -b 4 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix cnn_branch_SGD_473 -poly -branch cnn
done



#python -u train.py -net segnet -b 6 -lr 0.01 -e 2000 -dataset Glas -gpu -download
#python -u train.py -net segnet -b 6 -lr 0.01 -e 2000 -dataset Glas -gpu -download
#python -u train.py -net segnet -b 6 -lr 0.01 -e 2000 -dataset Glas -gpu -download
