# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

# import mmcv
# import numpy as np
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.logger.tensorboard import TensorboardLoggerHook
import torch
# from mmcv.runner.hooks.checkpoint import CheckpointHook
# from mmcv.runner.hooks.logger.wandb import WandbLoggerHook

# from mmseg.core import DistEvalHook, EvalHook



@HOOKS.register_module()
class MyTensorboardLoggerHook(TensorboardLoggerHook):

    #def is_hist(self, val):
    #    if isinstance(val, torch.Tensor):
    #        if val.numel > 1:
    #            return True

    #    return False

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))

            #elif self.is_hist(val):
            #    self.writter.add_histogram(tag, val, self.get_iter(runner))

            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))


        # print('hist_param' in runner.outputs)
        # print(runner.outputs['hist_param'])
        # print(runner.outputs.keys())

        if 'hist_param' in runner.outputs:
            for tag, val in runner.outputs['hist_param'].items():

                self.writer.add_histogram(tag, val, self.get_iter(runner))

            runner.logger.info(
                            f'Visualizaing param distributions at {runner.iter + 1} iterations')

            runner.outputs['hist_param'] = None