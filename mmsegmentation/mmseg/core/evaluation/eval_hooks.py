# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    # greater_keys = ['mIoU', 'mAcc', 'aAcc', 'cc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None

        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

        self._best_results = dict()


        for metric in ['all_objF1', 'all_objDice', 'testA_objF1', 'testA_objDice', 'testB_objF1', 'testB_objDice']:
            self.greater_keys.append(metric)
            self._best_results.update({metric : 0})

        for metric in ['all_objHausdorff', 'testA_objHausdorff', 'testB_objHausdorff']:
            self.less_keys.append(metric)
            self._best_results.update({metric : 0})

        # print(self.greater_keys)
        # print(self.less_keys)
        # import sys; sys.exit()
        # runner.meta['hook_msgs']['_best_score'] = self._best_results

        # self.

        self._best_score = {
            'all_objF1' : 0,
            'all_objDice' : 0,
            'all_objHausdorff' : 10000,
            'testA_objF1' : 0,
            'testA_objDice' : 0,
            'testA_objHausdorff' : 10000,
            'testB_objF1' : 0,
            'testB_objDice' : 0,
            'testB_objHausdorff' : 10000,
        }

        self._best_ckpt = {
            'all_objF1' : None,
            'all_objDice' : None,
            'all_objHausdorff' : None,
            'testA_objF1' : None,
            'testA_objDice' : None,
            'testA_objHausdorff' : None,
            'testB_objF1' : None,
            'testB_objDice' : None,
            'testB_objHausdorff' : None,
        }
        # self.indicator = ''
        # self.score = 0


    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        # print(runner.meta.keys())
        # import sys; sys.exit()
        from mmseg.apis import single_gpu_test
        results = single_gpu_test(
            runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner)
            #print(self._best_score)


    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            # return eval_res[self.key_indicator]

        return None


    def _compare_multiple_metric(self, runner):
        indicator = ''
        score = 0

        for name, best_val in self._best_score.items():
            eval_val = runner.log_buffer.output[name]

            # eval_val = runner.outputs[name]
            # print(name)
            if 'Hausdorff' in name:
                if eval_val < best_val:
                                # runner.meta['hook_msgs']['_best_score'][name] = eval_val
                    # save = True
                    # self.indicator = name
                    # self.score = eval_val
                    indicator = name
                    self._best_score[name] = eval_val
                    score = eval_val
                    # print(indicator, name)

            else:
                if eval_val > best_val:
                                # runner.meta['hook_msgs']['_best_score'][name] = eval_val
                    # save = True
                    indicator = name
                    score = eval_val
                    self._best_score[name] = eval_val
                    # print(indicator, name)

        return indicator, score


    # def _save_ckpt(self, runner, key_score):
    def _save_ckpt(self, runner):
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

        # self.
        # best_score = runner.meta['hook_msgs'].get(
            # 'best_score', self.init_value_map[self.rule])

        best_indicator, best_score = self._compare_multiple_metric(runner)
        # if self.compare_func(key_score, best_score):
            # best_score = key_score
        if best_score and best_indicator:

            # if self.best_ckpt_path and self.file_client.isfile(
            if self._best_ckpt[best_indicator] and self.file_client.isfile(
                        # self.best_ckpt_path):
                        self._best_ckpt[best_indicator]):

                    #if best_indicator in self.best_ckpt_path:
                    # self.file_client.remove(self.best_ckpt_path)
                    self.file_client.remove(self._best_ckpt[best_indicator])
                    runner.logger.info(
                            # f'The previous best checkpoint {self.best_ckpt_path} was '
                            f'The previous best checkpoint {self._best_ckpt[best_indicator]} was '
                            'removed')

            runner.meta['hook_msgs']['best_score'] = best_score
            self.key_indicator = best_indicator
            #for name, best_val in self._best_score.items():
            #            #print(name, val)
            #            # print(runner.log_buffer.output)
            #            # eval_val = runner.log_buffer.output[name]
            #            eval_val = runner.outputs[name]
            #            if 'Hausdorff' in name:
            #                if eval_val < best_val:
            #                    # runner.meta['hook_msgs']['_best_score'][name] = eval_val
            #                    save = True
            #                    self.indicator = name
            #                    self.score = eval_val

            #            else:
            #                if eval_val > best_val:
            #                    # runner.meta['hook_msgs']['_best_score'][name] = eval_val
            #                    save = True
            #                    self.indicator = name
            #                    self.score = eval_val

            best_ckpt_name = f'best_{self.key_indicator}_{current}.pth'
            self.best_ckpt_path = self.file_client.join_path(
                self.out_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path

            self._best_ckpt[self.key_indicator] = self.best_ckpt_path

            runner.save_checkpoint(
                self.out_dir,
                filename_tmpl=best_ckpt_name,
                create_symlink=False)
            runner.logger.info(
                f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(
                f'Best {self.key_indicator} is {best_score:0.4f} '
                f'at {cur_time} {cur_type}.')

            runner.logger.info(
                'Best results are:{}'.format(self._best_score)
            )
    ##############################################################
    #def evaluate(self, runner, results):
    #    """Evaluate the results.

    #    Args:
    #        runner (:obj:`mmcv.Runner`): The underlined training runner.
    #        results (list): Output results.
    #    """
    #    eval_res = self.dataloader.dataset.evaluate(
    #        results, logger=runner.logger, **self.eval_kwargs)

    #    for name, val in eval_res.items():
    #        runner.log_buffer.output[name] = val

    #        ##########################
    #         #['all_objF1', 'all_objDice', 'testA_objF1', 'testA_objDice', 'testB_objF1', 'testB_objDice']
    #        # if name in self.greater_keys:
    #            # if self._best_results[name] < val:
    #                # self._best_results[name] = val

    #        # if name in self.less_keys:
    #            # if self._best_results[name] > val:
    #                # self._best_results[name] = val

    #    runner.log_buffer.ready = True

    #    #print(runner.log_buffer.output)
    #    # runner.meta['hook_msgs']['_eval_res'] = eval_res
    #    # import sys; sys.exit()

    #    if self.save_best is not None:
    #        # If the performance of model is pool, the `eval_res` may be an
    #        # empty dict and it will raise exception when `self.save_best` is
    #        # not None. More details at
    #        # https://github.com/open-mmlab/mmdetection/issues/6265.
    #        if not eval_res:
    #            warnings.warn(
    #                'Since `eval_res` is an empty dict, the behavior to save '
    #                'the best checkpoint will be skipped in this evaluation.')
    #            return None

    #        if self.key_indicator == 'auto':
    #            # infer from eval_results
    #            self._init_rule(self.rule, list(eval_res.keys())[0])
    #        return eval_res[self.key_indicator]

    #    return None
    ##############################################################

class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval
        self.latest_results = None
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval)
        self.latest_results = results
        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                # self._save_ckpt(runner, key_score)
                self._save_ckpt(runner)
