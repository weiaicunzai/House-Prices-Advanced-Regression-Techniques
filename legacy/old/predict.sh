export CUDA_VISIBLE_DEVICES=0
#for i in 1 2 3
#for i in 1 
#do
#python -u train.py -net segnet -b 6 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op xor -prefix baselinee1500lr0.01 -baseline
#python -u train.py -net segnet -b 6 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op xor -prefix e1500lr0.01a5opxor
#python -u train.py -net segnet -b 6 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 2 -op xor -prefix e1500lr0.01a2opxor
#python -u train.py -net segnet -b 6 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix e1500lr0.01a5opnone
#python -u train.py -net segnet -b 4 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix del
#python -u train.py -net transunet -b 16 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix del
#python -u train.py -net transunet -b 4 -lr 0.01 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix transunet_baseline_nopretrain
#python -u pretrain.py -net  transseg -b 4 -lr 0.001 -e 1500 -dataset Glas -gpu -download -alpha 5 -op none -prefix transseg_SGD_473_poly_backbone_resnet50_fine_tune -pretrain
python -u predict.py -net  transseg  -img '/data/by/datasets/original/CRC_Dataset/Patient_019_04_Normal.png' -weight  checkpoints/transseg_SGD_473_poly_backbone_resnet50_fine_tune_Sunday_11_April_2021_15h_52m_55s/1200-best.pth -pretrain
#done



#python -u train.py -net segnet -b 6 -lr 0.01 -e 2000 -dataset Glas -gpu -download
#python -u train.py -net segnet -b 6 -lr 0.01 -e 2000 -dataset Glas -gpu -download
#python -u train.py -net segnet -b 6 -lr 0.01 -e 2000 -dataset Glas -gpu -download
