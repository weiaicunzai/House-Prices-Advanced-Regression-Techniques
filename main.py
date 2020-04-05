
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor



from dataset.iiit5k_word import IIIT5KWord
from utils.collate_fn import PadBatch

from transforms import Resize


transforms = Compose([
    Resize(height=32),
    ToTensor()
])
iiit5kword = IIIT5KWord('IIIT5K', 'alphabet/alphabet.txt', 'train', transforms=transforms)

print(len(iiit5kword))

data_loader = DataLoader(
    iiit5kword,
    128,
    shuffle=True,
    collate_fn=PadBatch(),
    num_workers=4
)


import time

start = time.time()
count = 0
for i, batch in enumerate(data_loader):
    images, labels, lengths = batch
    #print(type(images), images.shape)
    #print(labels.shape)
    #print(lengths.shape)
    count += 128

    #if (i + 1) % 30 == 0:
    #    finish = time.time()
    #    print('time:', round(finish - start, 4))
    #    print('count:', count)
    #    print('avg:', count / (finish - start))


finish = time.time()
print('time:', round(finish - start, 4))
print('count:', count)
print('avg:', count / (finish - start))
