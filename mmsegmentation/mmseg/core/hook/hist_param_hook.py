# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import Hook
#import mmcv.runner.hooks as hooks
# from mmcv.runner.hooks.checkpoint import CheckpointHook
# from mmcv.runner.hooks.logger.wandb import WandbLoggerHook

# from mmseg.core import DistEvalHook, EvalHook



@HOOKS.register_module()
class HistParamHook(Hook):
    def __init__(self, interval, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval

    @master_only
    # def before_train_epoch(self, runner):
    def after_train_iter(self, runner):
        # runner.model

        if self.every_n_iters(
                runner, self.interval): 
                # or (self.save_last
                                        #    and self.is_last_iter(runner)):

            # print('hello???')
            hist_param = dict()
            for name, param in runner.model.named_parameters():
                layer, attr = osp.splitext(name)
                attr = attr[1:]
                #writer.add_histogram("{}/{}".format(layer, attr), param.detach().cpu().numpy(), n_iter)
                # writer.add_histogram("{}/{}".format(layer, attr), param, n_iter)
                hist_param.update({"{}/{}".format(layer, attr) : param})
                # runner.output['hist_param'].update()
            # print(type(runner.log_buffer))
            runner.outputs['hist_param'] = hist_param

        #else:
        #    runner.outputs['hist_param'] = None


    #def before_train_iter(self, runner):
    #    print('self.before_train_iter')

    #def after_train_iter(self, runner):
    #    print('self.after_train_iter')

    #def after_train_epoch(self, runner):
    #    print('self.after_train_epoch')