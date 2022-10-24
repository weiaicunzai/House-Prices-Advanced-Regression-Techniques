import os
import random
import shutil



path = '/data/smb/数据集/结直肠/病理学/Extended_CRC/Original_Images'
image_names = os.listdir(path)

num = len(image_names)
train_num = int(num * 0.8)
test_num = num - train_num
train = random.sample(image_names, k=train_num)
print(train)

test = []
for i in image_names:
    if i not in train:
        test.append(i)


with open('/data/smb/数据集/结直肠/病理学/Extended_CRC/train.txt', 'w') as f:
    for i in train:
        f.write(i + '\n')

with open('/data/smb/数据集/结直肠/病理学/Extended_CRC/test.txt', 'w') as f:
    for i in test:
        f.write(i + '\n')
path = '/data/smb/数据集/结直肠/病理学/Extended_CRC/Original_Images'
train_path = os.path.join('/data/smb/数据集/结直肠/病理学/Extended_CRC/', 'train')
with open('/data/smb/数据集/结直肠/病理学/Extended_CRC/train.txt', 'r') as f:
    for i in f.readlines():
        i = i.strip()
        img_fp = os.path.join(path, i)
        print(img_fp, '    ', train_path)
        #print(img_fp)
        #shutil.move(img_fp, train_path)

#path = '/data/smb/数据集/结直肠/病理学/Extended_CRC/'
#train_path = os.path.join(path, 'train')