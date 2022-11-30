import time

class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self.count = 0


    def __next__(self):
        start = time.time()
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self._dataloader)
            s1 = time.time()
            data = next(self.iter_loader)
            s2 = time.time()
            print(s2 - s1)

        finish = time.time()
        self.count += 1
        print('Iterloader: {:04f}'.format(finish-start), self.count)
        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self