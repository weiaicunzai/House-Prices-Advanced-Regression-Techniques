
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
     """warmup_training learning rate scheduler
     Args:
         optimizer: optimzier(e.g. SGD)
         total_iters: totoal_iters of warmup phase
     """
     def __init__(self, optimizer, total_iters, last_epoch=-1):
         self.total_iters = total_iters
         super().__init__(optimizer, last_epoch)
     def get_lr(self):
         """we will use the first m batches, and set the learning
         rate to base_lr * m / total_iters
         """
         return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class WarmUpWrapper:
    def __init__(self, warmuplr_scheduler, lr_scheduler):
        self.warmup = warmuplr_scheduler
        self.warmup_iter = self.warmup.total_iters
        self.lr_scheduler = lr_scheduler

    def step(self, iter_idx):
        """ iter_idx: training iter_idx """
        if iter_idx < self.warmup_iter:
            self.warmup.step()
        else:
            self.lr_scheduler.step()


# def warmup(lr_func):

#     def wrapper(self, ):
#         if self.total_iters

    # class WarmUp():

        #def __init__(self, optimizer, warmup=None, max_lr=None, **kwargs):
        #    self.lr_schuduler = cls(optimizer, **kwargs)
        #    self.warmup = warmup
        #    self.max_lr = max_lr


        #def get_lr(self):
