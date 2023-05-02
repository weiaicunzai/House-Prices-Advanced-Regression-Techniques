import os
import argparse
import re

import torch
import torch.nn as nn

import transforms
from conf import settings
import utils
from metric import eval_metrics
from train import evaluate
from conf import settings
#from dataset.camvid import CamVid
#from metrics import Metrics
#from model import UNet


def gen_trans(combo):
    trans = transforms.MultiScaleFlipAug(
            # img_ratios=[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0],
            # rescale=True,
            img_ratios=[1],
            # flip=True,
            #flip=False,
            #flip_direction=['horizontal', 'vertical', ],
            #flip_direction=['h', 'v'],
            #flip_direction=['h', 'v', 'hv', 'r90'],
            #flip_direction=['h', 'v', 'r90'], + 1
            #flip_direction=['none', 'h', 'v', 'hv'],
            flip_direction=combo,
            #flip_direction=['h', 'v', 'r90h'],
            #flip_direction=['h', 'v', 'r90v'],
            # flip_direction=['h', 'v', 'hv', 'r90', 'r90h', 'r90v', 'r90hv', 'none'],
            #flip_direction=['h', 'v'],
            #flip_direction=['horizontal'],
            # transforms=[
                # transforms.ToTensor(),
                # transforms.Normalize(settings.MEAN, settings.STD),
            # ]
            resize_to_multiple=False,
            #min_size=208,
            #min_size=None,
            #min_size=480,
            #min_size=1024,
            # min_size=crop_size[0],
            #min_size=None, # F1 0.8776095444114832, Dice:0.8882521941014574,
            #min_size=args.size,
            min_size=480,
            mean=settings.MEAN,
            std=settings.STD
        )


    return trans



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-weight', type=str, required=True,
                        help='weight file path')
    parser.add_argument('-dataset', type=str, default='Camvid', help='dataset name')
    parser.add_argument('-net', type=str, required=True, help='network name')
    parser.add_argument('-download', action='store_true', default=False)
    parser.add_argument('-b', type=int, default=1,
                        help='batch size for dataloader')
    parser.add_argument('-pretrain', action='store_true', default=False, help='pretrain data')
    parser.add_argument('-branch', type=str, default='hybird', help='dataset name')
    parser.add_argument('-imgset', type=str, default='train', help='dataset name')

    args = parser.parse_args()
    print(args)

    #m = re.search(r'[a-zA-Z]+_[0-9]+_[a-zA-z]+_[0-9]+h_[0-9]+m_[0-9]s','Thursday_21_January_2021_09h_24m_07s')
    #m = re.search(r'[a-zA-Z]+_[0-9]+_[a-zA-Z]+_[0-9]+h_[0-9]+m','Thursday_21_January_2021_09h_24m_07s')
    #print(m.group())
    checkpoint = os.path.basename(os.path.dirname(args.weight))

    # for  i in range(100, 800, 10):
        # args.size = i
    # if args.dataset == 'Glas':
            #test_dataloader = utils.data_loader(args, 'testA')
    test_dataloader = utils.data_loader(args, 'val')
    test_dataset = test_dataloader.dataset
    # print(test_dataset)
    # print(test_dataset.transforms)
    net = utils.get_model(args.net, 3, test_dataset.class_num, args=args)
    net.load_state_dict(torch.load(args.weight))
    net = net.cuda()
    print(args.weight)
    net.eval()
    #print('Glas testA')


    augs = ['h', 'v', 'hv', 'r90', 'r90h', 'r90v', 'r90hv', 'none']
    from itertools import combinations
    count = 0
    # transforms = test_dataset.transforms

    img_set = args.imgset
    if img_set == 'trainA':
        val_set = 'testB'

    if img_set == 'trainB':
        val_set = 'testA'

    if img_set == 'train':
        val_set = 'val'

    test_dataloader = utils.data_loader(args, val_set)
    dataset = test_dataloader.dataset
    print('dataset:', len(dataset))
    multiscale_collate = test_dataloader.collate_fn



    for i in range(1, len(augs) + 1):
    #     # print(i)
        for combo in list(combinations(augs, i)):
            count += 1

    #         # test_dataset.transforms.flip_direction = combo

    #         # transforms.flip_direction = combo
    #         #print(combo)
    #         trans = gen_trans(list(combo))
    #         test_dataset.transforms = gen_trans(list(combo))
    #         test_dataloader.dataset.transforms = trans
    #         print(id(test_dataset.transforms), id(test_dataloader.dataset.transforms), id(trans))


            #test_dataloader.dataset.transforms.flip_direction = ['none', 'hv']
            #test_dataloader.dataset.transforms.flip_direction = list(combo)
            #if count == 1:
            #    combo1 = ['none', 'hv']
            #else:
            #    combo1 = ['r90', 'r90hv']
            # test_dataloader = utils.data_loader(args, 'val')
            dataset.transforms.flip_direction = combo
            dataset.transforms.min_size = 768

            test_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=4, num_workers=4, shuffle=False, pin_memory=True, persistent_workers=True,
                collate_fn=multiscale_collate)

            print(len(test_dataloader))

            # print(combo1,  test_dataloader.dataset.transforms.flip_direction, id(test_dataloader.dataset.transforms.flip_direction))
            with torch.no_grad():
                    results = evaluate(net, test_dataloader, args, val_set)
                    print('test dataset transforms is: ', test_dataloader.dataset.transforms)
                    for key, values in results.items():
                        print('{}: F1 {}, Dice:{}, Haus:{}'.format(key, *values))
                    # print(id(test_dataloader.dataset))

                    #for img_metas in test_dataloader:
                    ##     #print(len(img_meta))
                    #     for img_meta in img_metas:
                    #         print(img_meta['flip'], img_meta['img_name'], id(test_dataloader.dataset.transforms.flip_direction))
                    ##         pass


            # if count == 3:
            #     import sys; sys.exit()






            #utils.test(
            #    net,
            #    test_dataloader,
            #    settings.IMAGE_SIZE,
            #    settings.SCALES,
            #    settings.BASE_SIZE,
            #    test_dataset.class_num,
            #    settings.MEAN,
            #    settings.STD,
            #    checkpoint
            #)

        #test_dataloader = utils.data_loader(args, 'testB')
        #test_dataset = test_dataloader.dataset
        #print('Glas testB')
        #with torch.no_grad():
        #    results = evaluate(net, test_dataloader, args)
        #    for key, values in results.items():
        #        print('{}: F1 {}, Dice:{}, Haus:{}'.format(key, *values))
            #utils.test(
            #    net,
            #    test_dataloader,
            #    settings.IMAGE_SIZE,
            #    settings.SCALES,
            #    settings.BASE_SIZE,
            #    test_dataset.class_num,
            #    settings.MEAN,
            #    settings.STD,
            #    checkpoint
            #)

        #utils.test(
        #    net,
        #    test_dataloader,
        #    settings.IMAGE_SIZE,
        #    [1],
        #    settings.BASE_SIZE,
        #    test_dataset.class_num,
        #    settings.MEAN,
        #    settings.STD
        #)

#    import random
#    random.seed(42)
#    val_dataloader = utils.data_loader(args, 'val')
#    val_dataset = val_dataloader.dataset
#    cls_names = val_dataset.class_names
#    ig_idx = val_dataset.ignore_index
#    iou = 0
#    all_acc = 0
#    acc = 0
#
#    ioud = 0
#    all_accd = 0
#    accd = 0
#    with torch.no_grad():
#        for img, label in val_dataloader:
#            img = img.cuda()
#            b = img.shape[0]
#            label = label.cuda()
#            pred = net(img)
#            pred = pred.argmax(dim=1)
#
#            tmp_all_acc, tmp_acc, tmp_iou = eval_metrics(
#                    pred.detach().cpu().numpy(),
#                    label.detach().cpu().numpy(),
#                    len(cls_names),
#                    ignore_index=ig_idx,
#                    metrics='mIoU',
#                    nan_to_num=-1
#            )
#            tmp_all_accd, tmp_accd, tmp_ioud = dd.eval_metrics(
#                label.detach(),
#                pred.detach(),
#                len(cls_names)
#            )
#
#
#            all_acc += tmp_all_acc * b
#            acc += tmp_acc * b
#            iou += tmp_iou * b
#
#            all_accd += tmp_all_accd * b
#            accd += tmp_accd * b
#            ioud += tmp_ioud * b
#
#        all_acc /= len(val_dataloader.dataset)
#        acc /= len(val_dataloader.dataset)
#        iou /= len(val_dataloader.dataset)

        #print('Iou for each class:')
        #utils.print_eval(cls_names, iou)
        #print('Acc for each class:')
        #utils.print_eval(cls_names, acc)
        #miou = sum(iou) / len(iou)
        #macc = sum(acc) / len(acc)
        #print('Mean acc {:.4f} Mean iou {:.4f}  All Pixel Acc {:.4f}'.format(macc, miou, all_acc))
    #valid_transforms = transforms.Compose([
    #    transforms.Resize(settings.IMAGE_SIZE),
    #    transforms.ToTensor(),
    #    transforms.Normalize(settings.MEAN, settings.STD)
    #])
        #all_accd /= len(val_dataloader.dataset)
        #accd /= len(val_dataloader.dataset)
        #ioud /= len(val_dataloader.dataset)

        #print(accd)
        #print(ioud)
        #print(all_accd)

    #valid_dataset = CamVid(
    #    settings.DATA_PATH,
    #    'val',
    #    valid_transforms
    #)

    #valid_loader = torch.utils.data.DataLoader(
    #    valid_dataset, batch_size=args.b, num_workers=4)

    #metrics = Metrics(valid_dataset.class_num, valid_dataset.ignore_index)

    #loss_fn = nn.CrossEntropyLoss()

    #net = UNet(3, valid_dataset.class_num)
    #net.load_state_dict(torch.load(args.weight))
    #net = net.cuda()

    #net.eval()
    #test_loss = 0
    #with torch.no_grad():
    #    for batch_idx, (images, masks) in enumerate(valid_loader):

    #        images = images.cuda()
    #        masks = masks.cuda()

    #        preds = net(images)

    #        loss = loss_fn(preds, masks)
    #        test_loss += loss.item()

    #        preds = preds.argmax(dim=1)
    #        preds = preds.view(-1).cpu().data.numpy()
    #        masks = masks.view(-1).cpu().data.numpy()
    #        metrics.add(preds, masks)

    #        print('iteration: {}, loss: {:.4f}'.format(batch_idx, loss))

    #test_loss = test_loss / len(valid_loader)
    #miou = metrics.iou()
    #precision = metrics.precision()
    #recall = metrics.recall()
    #metrics.clear()


    #print(('miou: {miou:.4f}, precision: {precision:.4f}, '
    #       'recall: {recall:.4f}, average loss: {loss:.4f}').format(
    #    miou=miou,
    #    precision=precision,
    #    recall=recall,
    #    loss=test_loss
    #))
