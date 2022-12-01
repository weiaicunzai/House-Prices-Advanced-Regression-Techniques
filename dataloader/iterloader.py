import time

class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)


    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        return self