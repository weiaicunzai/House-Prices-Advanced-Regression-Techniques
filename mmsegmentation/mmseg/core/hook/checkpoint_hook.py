# import os.path as osp
from typing import Optional

import mmcv
# import numpy as np
from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only, allreduce_params
from mmcv.runner.hooks.checkpoint import CheckpointHook
# from mmcv.runner.hooks import Hook









@HOOKS.register_module()
class MyCheckpointHook(CheckpointHook):

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 out_dir: Optional[str] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 sync_buffer: bool = False,
                 file_client_args: Optional[dict] = None,
                 **kwargs):
        
        super().__init__(
            interval=interval,
            by_epoch=by_epoch,
            save_optimizer=save_optimizer,
            out_dir=out_dir,
            max_keep_ckpts=max_keep_ckpts,
            save_last=save_last,
            sync_buffer=sync_buffer,
            file_client_args=file_client_args,
            **kwargs
        )
        # print(self.by_epoch)
        # print(self.args)
        # import sys; sys.exit()

        #self.interval = interval
        #self.by_epoch = by_epoch
        #self.save_optimizer = save_optimizer
        #self.out_dir = out_dir
        #self.max_keep_ckpts = max_keep_ckpts
        #self.save_last = save_last
        #self.args = kwargs
        #self.sync_buffer = sync_buffer
        #self.file_client_args = file_client_args

        # print(by_epoch)
        # import sys; sys.exit()

        self._best_score = { 
            'all_objF1' : 0,
            'all_objDice' : 0,
            'all_objHausdorff' : 0,
            'testA_objF1' : 0,
            'testA_objDice' : 0,
            'testA_objHausdorff' : 0,
            'testB_objF1' : 0,
            'testB_objDice' : 0,
            'testB_objHausdorff' : 0,
        } 
        self.indicator = ''
        self.score = 0
        # print('ccccccccccccccccccccc')
        # print(self.by_epoch)
        # import sys; sys.exit()

    #@master_only
    #def _save_checkpoint(self, runner):
    #    """Save the current checkpoint and delete unwanted checkpoint."""
    #    runner.save_checkpoint(
    #        self.out_dir, save_optimizer=self.save_optimizer, **self.args)
    #    if runner.meta is not None:
    #        if self.by_epoch:
    #            cur_ckpt_filename = self.args.get(
    #                'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
    #        else:
    #            cur_ckpt_filename = self.args.get(
    #                'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
    #        runner.meta.setdefault('hook_msgs', dict())
    #        runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
    #            self.out_dir, cur_ckpt_filename)
    #    # remove other checkpoints
    #    if self.max_keep_ckpts > 0:
    #        if self.by_epoch:
    #            name = 'epoch_{}.pth'
    #            current_ckpt = runner.epoch + 1
    #        else:
    #            name = 'iter_{}.pth'
    #            current_ckpt = runner.iter + 1
    #        redundant_ckpts = range(
    #            current_ckpt - self.max_keep_ckpts * self.interval, 0,
    #            -self.interval)
    #        filename_tmpl = self.args.get('filename_tmpl', name)
    #        for _step in redundant_ckpts:
    #            ckpt_path = self.file_client.join_path(
    #                self.out_dir, filename_tmpl.format(_step))
    #            if self.file_client.isfile(ckpt_path):
    #                self.file_client.remove(ckpt_path)
    #            else:
    #                break

    def after_train_iter(self, runner):
        # for i in dir(runner):
            # print(i)

        # print(self.by_epoch)
        # print('............My checkpoint function!!!!!!!', runner.iter, runner.max_iters)



        if self.by_epoch:
            return

        save = False
        # indicator = ''
        # score = 0
        #for name, val in _best_score.

        # only saving best checkpoints after half of the iterations
        #if runner.iter < runner.max_iters * 0.5:
        # if runner.iter == -200:
            # print(runner.iter, runner.iters)

            # self.best_indicator = ''
            #for name, best_val in self._best_score.items():
            #    #print(name, val)
            #    # print(runner.log_buffer.output)
            #    eval_val = runner.log_buffer.output[name] 
            #    if 'Hausdorff' in name:
            #        if eval_val < best_val:
            #            # runner.meta['hook_msgs']['_best_score'][name] = eval_val
            #            save = True
            #            self.indicator = name
            #            self.score = eval_val

            #    else:
            #        if eval_val > best_val:
            #            # runner.meta['hook_msgs']['_best_score'][name] = eval_val
            #            save = True
            #            self.indicator = name
            #            self.score = eval_val

            ## print(self.args)
            ## import sys; sys.exit()
            #if save:
            #    self._save_best_ckpt(runner)

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training

        # print(runner.log_buffer.output, runner.iter)

        if self.every_n_iters(
            runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):

            for name, best_val in self._best_score.items():
                        #print(name, val)
                        # print(runner.log_buffer.output)
                        # eval_val = runner.log_buffer.output[name] 
                        eval_val = runner.outputs[name] 
                        if 'Hausdorff' in name:
                            if eval_val < best_val:
                                # runner.meta['hook_msgs']['_best_score'][name] = eval_val
                                save = True
                                self.indicator = name
                                self.score = eval_val

                        else:
                            if eval_val > best_val:
                                # runner.meta['hook_msgs']['_best_score'][name] = eval_val
                                save = True
                                self.indicator = name
                                self.score = eval_val

                    # print(self.args)
                    # import sys; sys.exit()
            if save:
                self._save_best_ckpt(runner)
                return



            runner.logger.info(
                            f'Saving checkpoint at {runner.iter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())


            self._save_checkpoint(runner)


        # if runner.meta['hook_msgs']['']
        # _eval_res = runner.meta['hook_msgs']['_eval_res'] 
        # _best_score = runner.meta['hook_msgs']['_best_score']





    @master_only
    def _save_best_ckpt(self, runner, key_indicator, ):
        """Save the best checkpoint.

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        #best_score = runner.meta['hook_msgs'].get(
        #    'best_score', self.init_value_map[self.rule])
        #if self.compare_func(key_score, best_score):
        #    best_score = key_score
        #    runner.meta['hook_msgs']['best_score'] = best_score

        #    if self.best_ckpt_path and self.file_client.isfile(
        #            self.best_ckpt_path):
        #        self.file_client.remove(self.best_ckpt_path)
        #        runner.logger.info(
        #            f'The previous best checkpoint {self.best_ckpt_path} was '
        #            'removed')

            best_ckpt_name = f'best_{self.indicator}_{current}.pth'
            best_ckpt_path = self.file_client.join_path(
                runner.out_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = best_ckpt_path

            runner.save_checkpoint(
                runner.out_dir,
                filename_tmpl=best_ckpt_name,
                create_symlink=False)
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(
                f'Best {self.indicator} is {self.score:0.4f} '
                f'at {cur_time} {cur_type}.')

    #@master_only
    #def _save_best_checkpoint(self, runner):
    #    """Save the current checkpoint and delete unwanted checkpoint."""

    #    self.args['filename_tmpl'] = 'iter_best_{}.path'
    #    if runner.meta is not None:
    #        if self.by_epoch:
    #            cur_ckpt_filename = self.args.get(
    #                'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
    #        else:
    #            cur_ckpt_filename = self.args.get(
    #                'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
    #        runner.meta.setdefault('hook_msgs', dict())
    #        runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
    #            self.out_dir, cur_ckpt_filename)

    #    runner.save_checkpoint(
    #        self.out_dir, save_optimizer=self.save_optimizer, **self.args)

    #    # remove other checkpoints
    #    if self.max_keep_ckpts > 0:
    #        if self.by_epoch:
    #            name = 'epoch_{}.pth'
    #            current_ckpt = runner.epoch + 1
    #        else:
    #            name = 'iter_{}.pth'
    #            current_ckpt = runner.iter + 1
    #        redundant_ckpts = range(
    #            current_ckpt - self.max_keep_ckpts * self.interval, 0,
    #            -self.interval)
    #        filename_tmpl = self.args.get('filename_tmpl', name)
    #        for _step in redundant_ckpts:
    #            ckpt_path = self.file_client.join_path(
    #                self.out_dir, filename_tmpl.format(_step))
    #            if self.file_client.isfile(ckpt_path):
    #                self.file_client.remove(ckpt_path)
    #            else:
    #                break