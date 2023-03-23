import os
import sys
sys.path.append(os.getcwd())
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter


from train import evaluate
import utils
from conf import settings





def test_evaluate():

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.gpu = True
    args.net = 'tgt'
    args.fp16 = True
    args.b = 4
    args.dataset = 'crag'
    args.pretrain = False
    args.iter = 1000
    args.prefix = 'unet_branch_SGD_473'
    args.eval_iter = 30


    val_loader = utils.data_loader(args, 'val')
    test_dataset = val_loader.dataset
    net = utils.get_model(args.net, 3, test_dataset.class_num, args=args)
    ckpt_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/checkpoints/tri_graph_Wednesday_15_March_2023_23h_47m_53s/iter_39999.pt'

    # glas+crag+rings+densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_sin_rings_densecl/latest.pth'
    # glas+crag+sin+densecl
    #ckpt_path = '/data/hdd1/by/mmselfsup/work_dir_glas_crag_sin/latest.pth'
    print('Loading pretrained checkpoint from {}'.format(ckpt_path))
    #new_state_dict = utils.on_load_checkpoint(net.state_dict(), torch.load(ckpt_path)['state_dict'])
    new_state_dict = utils.on_load_checkpoint(net.state_dict(), torch.load(ckpt_path))
    net.load_state_dict(new_state_dict)
    print('Done!')
    if args.gpu:
        net = net.cuda()

    # evaluate(net, val_loader, args)



# def dummy_evaluate():

    root_path = os.path.dirname(os.path.abspath(__file__))

    log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)
    print('saving tensorboard log into {}'.format(log_dir))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    ckpt_path = os.path.join(
        root_path, settings.CHECKPOINT_FOLDER, args.prefix + '_' + settings.TIME_NOW)
    # log_dir = os.path.join(root_path, settings.LOG_FOLDER, args.prefix + '_' +settings.TIME_NOW)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_manager = utils.CheckPointManager(ckpt_path, max_keep_ckpts=5)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr * args.scale, momentum=0.9, weight_decay=5e-4)
    # iter_per_epoch = len(train_dataset) / args.b

    # max_iter = args.e * len(train_loader)
    total_iter = args.iter
    warmup_iter = int(args.iter * 0.1)


    net.train()


    # scaler = GradScaler()

    total_metrics = ['total_F1', 'total_Dice', 'total_Haus']
    testA_metrics = ['testA_F1', 'testA_Dice', 'testA_Haus']
    testB_metrics = ['testB_F1', 'testB_Dice', 'testB_Haus']

    vis_idx = 4
    # train_t = time.time()
    # for iter_idx, (images, masks, weight_maps) in enumerate(train_iterloader):
    for iter_idx in range(1000):


        # print log
        #if args.eval_iter % (iter_idx + 1) == 0:
        # print(args.eval_iter)
        if (iter_idx + 1) % args.eval_iter == 0:

            # evaluate()


            net.eval()
            print('evaluating.........')
            #total, testA, testB = evaluate(net, val_loader, args)
            results = evaluate(net, val_loader, args)

            #print('total: F1 {}, Dice:{}, Haus:{}'.format(*total))

            for key, values in results.items():
                print('{}: F1 {}, Dice:{}, Haus:{}'.format(key, *values))



            utils.visualize_metric(writer,
                #['total_F1', 'total_Dice', 'total_Haus'], total, iter_idx)
                #total_metrics, total, iter_idx)
                total_metrics, results['total'], iter_idx)

            if args.dataset == 'Glas':
                utils.visualize_metric(writer,
                    #['testA_F1', 'testA_Dice', 'testA_Haus'], testA, iter_idx)
                    testA_metrics, results['testA'], iter_idx)

                utils.visualize_metric(writer,
                    #['testB_F1', 'testB_Dice', 'testB_Haus'], testB, iter_idx)
                    testB_metrics, results['testB'], iter_idx)




            utils.visualize_param_hist(writer, net, iter_idx)


            ckpt_manager.save_ckp_iter(net, iter_idx)
            ckpt_manager.save_best(net, total_metrics,
                results['total'], iter_idx)

            if args.dataset == 'Glas':
                ckpt_manager.save_best(net, testA_metrics,
                    results['testA'], iter_idx)
                ckpt_manager.save_best(net, testB_metrics,
                    results['testB'], iter_idx)

            print('best value:', ckpt_manager.best_value)
            net.train()


        # train_t = time.time()

        if total_iter <= iter_idx:
            break


test_evaluate()