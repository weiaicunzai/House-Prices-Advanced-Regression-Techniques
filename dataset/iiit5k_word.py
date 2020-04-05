
import os
#import sys
#sys.path.append(os.getcwd())

import cv2
import scipy.io
import numpy as np

import torch
#from torchvision.transforms import Compose, ToTensor

from torch.utils.data import Dataset
#import torch.nn.functional as F


class IIIT5KWord(Dataset):

    def __init__(self, data_dir, alphabet_fp, phase, transforms=None):
        assert phase in ['train', 'test']

        with open(alphabet_fp) as f:
            self.alphabet = f.read()

        alphabet_length = len(self.alphabet)
        self.alphabet = {
            k : v for k, v in zip(self.alphabet, range(alphabet_length))
        }

        self.label_key = '{}data'.format(phase)
        data_path = os.path.join(data_dir, '{}.mat'.format(self.label_key))
        self.data = scipy.io.loadmat(data_path)
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data[self.label_key][0])

    def __getitem__(self, index):
        image_name = self.data[self.label_key][0][index][0][0]
        #print(image_name.tostring().decode())
        text  = self.data[self.label_key][0][index][1][0]

        image_path = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_path)

        if self.transforms:
            image = self.transforms(image)

        label = []
        for character in text:
            label.append(self.alphabet[character])



        #image_name, label = self.label[self.label_key][0][0:2]

        return image, label, len(label)





#transform
#transform = transforms.ToTensor()
#transforms = Compose([
#    Resize(height=32),
#    ToTensor()
#])
#iiit5kword = IIIT5KWord('IIIT5K', 'alphabet/alphabet.txt', 'train', transforms=transforms)
#
#print(len(iiit5kword))
#
#data_loader = DataLoader(
#    iiit5kword,
#    32,
#    shuffle=True,
#    collate_fn=PadBatch()
#)
#
#
#import time
#
#start = time.time()
#count = 0
#for i, batch in enumerate(data_loader):
#    images, labels, lengths = batch
#    print(type(images), images.shape)
#    print(labels.shape)
#    print(lengths.shape)
#    count += 32
#
#    if (i + 1) % 20 == 0:
#        finish = time.time()
#        print('time:', round(finish - start, 4))
#        print('count:', count)
#        print('avg:', count / (finish - start))
#
#
#finish = time.time()
#print('time:', round(finish - start, 4))
#print('count:', count)
#print('avg:', count / (finish - start))
#
#
##alphabet = set()
##for i in range(len(iiit5kword)):
##    image, label = iiit5kword[i]
##
##    for c in label:
##        alphabet.add(c)
##
##print(''.join(list(alphabet)))
##for
##image, label = iiit5kword[44]
##cv2.imshow(label, image)
##cv2.waitKey(0)
##print(image.shape, label)
##print(iiit5kword[44])
#