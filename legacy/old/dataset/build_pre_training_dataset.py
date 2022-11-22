import os
import re
import glob

import lmdb
from PIL import Image
import imagesize


Train_DATASET = {
    #'CRAG' : ['/data/by/datasets/original/CRAG/train/Images', 'train', 'train_[0-9]+.png'], #check
    'CRAG' : ['/data/smb/数据集/结直肠/病理学/CRAG/CRAG/train/Images', 'train', 'train_[0-9]+.png'], #check
    #'Extended_CRC' : ['/data/smb/数据集/结直肠/病理学/Extended_CRC/train', 'train', '.*.png'], # check
    'CRC' : ['/data/by/datasets/original/CRC_Dataset', 'train', '.*.png'], #check
    'Glas' : ['/data/by/datasets/original/Warwick QU Dataset (Released 2016_07_08)', 'train', 'train_[0-9]+.bmp'], #check

    'CoNseP' : ['/data/by/datasets/original/CoNSep/CoNSeP/Train/Images', 'train', '.*.png'] # 1000x1000 #check
    #'' : '/data/by/datasets/original/CoNSep/CoNSeP/Train/Images' # 1000x1000
}

Test_DATASET = {
    'CRAG' : ['/data/by/datasets/original/CRAG/valid/Images', 'test', 'test_[0-9]+.png'], # check
    'Glas' : ['/data/by/datasets/original/Warwick QU Dataset (Released 2016_07_08)', 'test', 'test[A|B]_[0-9]+.bmp'], # check
    'CoNseP' : ['/data/by/datasets/original/CoNSep/CoNSeP/Test/Images', 'test', 'test_[0-9]+.png'], # 1000x1000  # check
    'CRC' : ['/data/by/datasets/original/test', 'test', '.*.png'], #check
    #'Extended_CRC' : ['/data/smb/数据集/结直肠/病理学/Extended_CRC/test', 'test', '.*.png'], # check
}

class DataSet:
    def __init__(self, name, data_path, image_set, regex_str):
        self.data_path_abs = data_path
        self.data_names = name
        self.image_set = image_set
        self.image_shape = ()
        self.total_area = 0
        self.image_types = ['png', 'bmp']
        self.regex_str = regex_str
        self.image_fp = []
        self._stats()
        print(self.total_area)
        print(self.image_shape)

    def _stats(self):
        search_path = os.path.join(self.data_path_abs, '**', '*')
        size = set()
        for i in glob.iglob(search_path, recursive=True):
            if not re.search(self.regex_str, i):
                continue
            print(i)
            img = Image.open(i)
            size.add(img.size)
            area = img.size[0] * img.size[1]
            self.total_area += area
            self.image_fp.append(i)
            print(len(self.image_fp))

        self.image_shape = tuple(size)

def construct_dataset(config):
    datasets = []
    for key, value in config.items():
        #print(key, value)
        data = DataSet(key, *value)
        datasets.append(data)

    return datasets

#print()
#construct_dataset(Train_DATASET)
#construct_dataset(Test_DATASET)
#data = DataSet('Glas', '/data/by/datasets/original/Warwick QU Dataset (Released 2016_07_08)', 'train', 'train_[0-9]+.bmp')
#data = DataSet('Glas', '/data/by/datasets/original/Warwick QU Dataset (Released 2016_07_08)', 'test', 'test[A|B]_[0-9]+.bmp')

#import base64
#
lmdb_fp = '/data/by/House-Prices-Advanced-Regression-Techniques/data/pre_training/test'
##os.mkdir(lmdb_fp)
if not os.path.exists(lmdb_fp):
    os.mkdirs(lmdb_fp)
db_size = 1 << 40
#
env = lmdb.open(lmdb_fp, map_size=db_size)
#
#
#import cv2
#import io
from PIL import Image
import numpy as np
##count = 0
##with env.begin() as txn:
##    for k, v in txn.cursor():
##         img = Image.open(io.BytesIO(v))
##         print(img.size)
##         count += 1
####         #k = k.decode()
##         print(k)
##         img.save('tmp/{}.jpg'.format(count))
#
def read_lmdb(env):
    with env.begin() as txn:
        count = 0
        for k, v in txn.cursor():
            img = Image.open(io.BytesIO(v))
            print(img.size)
            count += 1
            print(k)
            img.save('tmp/{}.jpg'.format(count))

def write_lmdb(env, datasets):
#        #img = np.fromstring(v)
#        file_bytes = np.asarray(bytearray(v), dtype=np.uint8)
#        print(file_bytes.shape)
#        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#
#        #img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#        print(img.shape)



    with env.begin(write=True) as txn:
        for dataset in datasets:
            for img_fp in dataset.image_fp:
                with open(img_fp, 'rb') as f:
#                        #img_str = base64.encodebytes(f.read()).decode('utf-8')
#                        #print(img_fp, len(img_str))
                        #image_name = img_fp.encode('utf-8')
                        #txn.put(image_name, )
####
                        img_bytes = f.read()
##                        print(type(img_bytes))
                        print(img_fp.encode())
                        txn.put(img_fp.encode(), img_bytes)
##                        image = Image.open(io.BytesIO(img_bytes))
##                        print(image.size)
#                       #print(type(img_str))

datasets = construct_dataset(Test_DATASET)
write_lmdb(env, datasets)