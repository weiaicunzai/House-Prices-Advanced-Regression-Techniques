import os
import argparse
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
#from model import UNet
from conf import settings
from PIL import Image
import utils


def unnorm(image):
    image = image.squeeze(0)
    print(image.shape)
    #print(image.shape).permute(2, 0, 1)
    image = image.mul_(std[:, None, None]).add_(mean[:, None, None]) * 255
    image = image.permute(1, 2, 0)
    print(image.shape)

    image = image.cpu().numpy()

    return image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-weight', type=str, required=True,
                        help='weight file path')
    #parser.add_argument('-dataset', type=str, default='Camvid', help='dataset name')
    parser.add_argument('-net', type=str, required=True, help='network name')
    parser.add_argument('-download', action='store_true', default=False)
    parser.add_argument('-b', type=int, default=1,
                        help='batch size for dataloader')
    parser.add_argument('-pretrain', action='store_true', default=False, help='pretrain data')

    args = parser.add_argument('-img', type=str, required=True,
                        help='image path to predict')

    #args = parser.add_argument('-weight', type=str, required=True,
    #                    help='weight file path')

    #args = parser.add_argument('-c', type=int, default=32,
    #                    help='class number')

    args = parser.parse_args()

    data_transforms = transforms.Compose([
        transforms.CenterCrop(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.MEAN, settings.STD)
    ])

    data_transforms = utils.pretrain_test_transforms()

    args.img = '/data/by/datasets/original/CRC_Dataset/Patient_019_04_Normal.png'

    src = cv2.imread(args.img)
    #image = Image.fromarray(src)
    #image = cv2.imread(args., -1)
    image = data_transforms(src)
    image = image.unsqueeze(0)
    image = image.cuda()



    net = utils.get_model(args.net, 3, 2, args=args)

    a = torch.load(args.weight)
    net.load_state_dict(torch.load(args.weight))
    net = net.cuda()

    net.eval()

    mean = torch.tensor(settings.MEAN, dtype=torch.float32).cuda()
    std = torch.tensor(settings.STD, dtype=torch.float32).cuda()
    with torch.no_grad():

        preds = net(image)

        #preds = torch.argmax(preds, dim=1)
        #preds = preds.cpu().data.numpy()
        #preds = preds.squeeze(0)
        #preds = preds.argmax(dim=1)
        preds = unnorm(preds)
        image = unnorm(image)


    #image = image.squeeze(0)
    #print(image.shape)
    ##print(image.shape).permute(2, 0, 1)
    #image = image.mul_(std[:, None, None]).add_(mean[:, None, None]) * 255
    #image = image.permute(1, 2, 0)
    #print(image.shape)

    #image = image.cpu().numpy()
    #image = image.

    #preds = cv2.resize(preds, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('src.jpg', image)
    cv2.imwrite('predict.jpg', preds)