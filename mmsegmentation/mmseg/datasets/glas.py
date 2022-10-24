
# import os
# import zipfile
# import shutil
# import glob
# import re
import warnings
from prettytable import PrettyTable


# import cv2
# import torch.nn as nn
# from torch.utils.data import Dataset
# from torchvision.datasets.utils import download_url
import numpy as np

import mmcv
from mmcv.utils import print_log
from mmseg.core.evaluation.metrics import gland_accuracy_object_level
# from metric import gland_accuracy_object_level
from mmseg.utils import get_root_logger

from .builder import DATASETS
from .custom import CustomDataset



@DATASETS.register_module()
#class GlaSDataset(Dataset):
class GlaSDataset(CustomDataset):

    CLASSES = ('background', 'gland')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    #def __init__(self, path, image_set, transforms=None, download=False):
    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.bmp',
            seg_map_suffix='_anno.bmp',
            reduce_zero_label=False,
            **kwargs
        )

        self.label_map = {
            i: 1 for i in range(1, 33)
        }


        # print(self.label_map)
        # import sys; sys.exit()


        #url = 'https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip'
        #file_name = 'warwick_qu_dataset_released_2016_07_08.zip'
        #md5 = '495b2a9f3d694545fbec06673fb3f40f'

        #if download:
        #    download_url(url, path, file_name, md5=md5)

        ## self.class_names = ['background', 'gland']
        ## self.ignore_index = -100
        ## self.class_num = len(self.class_names)

        #data_folder = os.path.join(path, 'Warwick QU Dataset (Released 2016_07_08)')
        #if not os.path.exists(data_folder):
        #    if not os.path.exists(os.path.join(path, file_name)):
        #        raise RuntimeError('Dataset not found or corrupted.' +
        #                           ' You can use download=True to download it')

        #    with zipfile.ZipFile(os.path.join(path, file_name), "r") as f:
        #        f.extractall(path=path)

        # self.images = []
        # self.labels = []
        # search_path = os.path.join(data_folder, '**', '*.bmp')
        #image_re = re.escape(image_set + '_[0-9]+\.bmp')
        #image_re = image_set + '_[0-9]+\.bmp'
        #label_re = re.escape(image_set + '_[0-9]+_anno\.bmp')

        #if image_set not in ['train', 'testA', 'testB', 'val']:
        #    raise ValueError('wrong image_set argument')
        #label_re = image_set + '_[0-9]+_' + 'anno' + '\.bmp'
        #if image_set == 'val':
        #    label_re = 'test[A|B]' +  '_[0-9]+_' + 'anno' + '\.bmp'

        #for bmp in glob.iglob(search_path, recursive=True):
        #    if re.search(label_re, bmp):
        #        self.labels.append(cv2.imread(bmp, -1))
        #        bmp = bmp.replace('_anno', '')
        #    #elif re.search(image_re, bmp):
        #        self.images.append(cv2.imread(bmp, -1))

        #assert len(self.images) == len(self.labels)
        #self.transforms = transforms
        #self.mean = (0.7851387990848604, 0.5111793462233759, 0.787433705481764)
        #self.std = (0.13057256006459803, 0.24522816688399154, 0.16553457394913107)
        #self.image_set = image_set

        #print(self.img_infos)

        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

        # print(len(self.img_infos))
        # for i in self.img_infos:
            # print(i)

        # print(self.pipeline)
        # self.metrics = 
        self.metrics = ['objF1', 'objDice', 'objHausdorff']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            # yield results['gt_semantic_seg']
            yield results

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        # print(self.custom_classes)
        if self.label_map:
            results['label_map'] = self.label_map

        # print(res)
        # import sys; sys.exit()

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                             split):
            """Load annotation from directory.

            Args:
                img_dir (str): Path to image directory
                img_suffix (str): Suffix of images.
                ann_dir (str|None): Path to annotation directory.
                seg_map_suffix (str|None): Suffix of segmentation maps.
                split (str|None): Split txt file. If split is specified, only file
                    with suffix in the splits will be loaded. Otherwise, all images
                    in img_dir/ann_dir will be loaded. Default: None

            Returns:
                list[dict]: All image info of dataset.
            """

            img_infos = []
            if split is not None:
                lines = mmcv.list_from_file(
                    split, file_client_args=self.file_client_args)

                lines = [line for line in lines if '_anno' not in line]
                #lines = []
                #for line in lines_tmp:
                #    if '_anno' in line:
                #        continue
                #    lines.append(line)

                for line in lines:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
            else:
                for img in self.file_client.list_dir_or_file(
                        dir_path=img_dir,
                        list_dir=False,
                        suffix=img_suffix,
                        recursive=True):

                    if '_anno' in img:
                        continue

                    img_info = dict(filename=img)
                    if ann_dir is not None:
                        seg_map = img.replace(img_suffix, seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
                img_infos = sorted(img_infos, key=lambda x: x['filename'])

            print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
            return img_infos

    # def _evaluate_glas(self, )

    def _evaluate_glas(self, results, anno_results, logger, print_sample=False):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: GlaS evaluation results.
        """
        #try:
        #    import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        #except ImportError:
        #    raise ImportError('Please run "pip install cityscapesscripts" to '
        #                      'install cityscapesscripts first.')
        msg = 'Evaluating in GlaS Dataset'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        # result_dir = imgfile_prefix

        eval_results = dict()
        # print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        # CSEval.args.evalInstLevelScore = True
        # CSEval.args.predictionPath = osp.abspath(result_dir)
        # CSEval.args.evalPixelAccuracy = True
        # CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        #for seg_map in mmcv.scandir(
        #        self.ann_dir, 'gtFine_labelIds.png', recursive=True):
        #    seg_map_list.append(osp.join(self.ann_dir, seg_map))
        #    pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        #eval_results.update(
        #    CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        # f1, dice, hausdorff
        partA = [0, 0, 0]
        partB = [0, 0, 0]
        count_sampleA = 0
        count_sampleB = 0
        # metrics = ['objF1', 'objDice', 'objHausdorff']
        for result, anno_result in zip(results, anno_results):
            # print(result)
            # print(anno_result)
            # print()
            gt_seg_map = anno_result['gt_semantic_seg']
            pred_seg_map = result
            recall, precision, obj_f1, obj_dice, obj_iou,  obj_hausdorff = \
                gland_accuracy_object_level(pred_seg_map, gt_seg_map)

            anno_filename = anno_result['ann_info']['seg_map']
            sample_result = obj_f1, obj_dice, obj_hausdorff

            if print_sample: 
                msg = '{}, iou: {:.6f}, f1: {:.6f}, recall: {:.6f}, precision: {:.6f}, dice: {:.6f}, hausdorff: {:.6f}'.format(
                    anno_filename,
                    obj_iou,
                    obj_f1,
                    recall,
                    precision,
                    obj_dice,
                    obj_hausdorff
                )
                print_log(msg)

            if 'testA' in anno_filename:
                count_sampleA += 1
                partA = [sum(m) for m in zip(partA, sample_result)]
                #partA[0] += obj_f1
                #partA[1] += obj_dice
                #partA[2] += obj_hausdorff

            elif 'testB' in anno_filename:
                count_sampleB += 1
                partB = [sum(m) for m in zip(partB, sample_result)]
                #partB[0] += obj_f1
                #partB[1] += obj_dice
                #partB[2] += obj_hausdorff
            else:
                raise ValueError('filenames of GlaS dataset should contain \
                    one of these "testA", "testB". ')

        total = [sum(m) for m in zip(partA, partB)]
        total = [m / (count_sampleA + count_sampleB) for m in total]

        eval_results.update(zip(
            ['all_{}'.format(metric) for metric in self.metrics],
            total
        ))

        partA = [m / count_sampleA for m in partA]
        eval_results.update(zip(
            ['testA_{}'.format(metric) for metric in self.metrics],
            partA
        ))

        partB = [m / count_sampleB for m in partB]
        eval_results.update(zip(
            ['testB_{}'.format(metric) for metric in self.metrics],
            partB
        ))


        # print(eval_results)

        # import sys; sys.exit()

            #print(result, gt_seg_map.shape)


        return eval_results


    # def _evaluate_glas
    def evaluate(self,
                 results,
                 metric='glas',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset
        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        # allowed_metrics = ['objIoU', 'objHausdorff', 'objFscore']
        allowed_metrics = ['glas']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        # test a list of files
        #print(results)
        #import sys; sys.exit()
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):

            # print(gt_seg_maps, 11111)
            # import sys; sys.exit()
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            # num_classes = len(self.CLASSES)
            #zip(results, gt_seg_maps)
            #if 'print_sample' in kwargs:
            print_sample = kwargs.pop('print_sample', None)
            eval_results = self._evaluate_glas(results, gt_seg_maps, logger, print_sample=print_sample)

            # import sys; sys.exit()
            #ret_metrics = eval_metrics(
            #    results,
            #    gt_seg_maps,
            #    num_classes,
            #    self.ignore_index,
            #    metric,
            #    label_map=dict(),
            #    reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        #else:
        #    ret_metrics = pre_eval_to_metrics(results, metric)
        # Because dataset.CLASSES is required for per-eval.
        #if self.CLASSES is None:
        #    class_names = tuple(range(num_classes))
        #else:
        #    class_names = self.CLASSES
        ## summary table
        #ret_metrics_summary = OrderedDict({
        #    ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        #    for ret_metric, ret_metric_value in ret_metrics.items()
        #})
        ## each class table
        #ret_metrics.pop('aAcc', None)
        #ret_metrics_class = OrderedDict({
        #    ret_metric: np.round(ret_metric_value * 100, 2)
        #    for ret_metric, ret_metric_value in ret_metrics.items()
        #})
        #ret_metrics_class.update({'Class': class_names})
        #ret_metrics_class.move_to_end('Class', last=False)
        ## for logger
        #class_table_data = PrettyTable()
        #for key, val in ret_metrics_class.items():
        #    class_table_data.add_column(key, val)
        summary_table_data = PrettyTable()
        # for key, val in ret_metrics_summary.items():
        for key, val in eval_results.items():
            #if key == 'aAcc':
            #    summary_table_data.add_column(key, [val])
            #else:
                # summary_table_data.add_column('m' + key, [val])
            summary_table_data.add_column(key, [val])
        #print_log('per class results:', logger)
        #print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)
        # each metric dict
        # for key, value in ret_metrics_summary.items():
        #for key, value in eval_results.items():
        #    if key == 'aAcc':
        #        eval_results[key] = value / 100.0
        #    else:
        #        eval_results['m' + key] = value / 100.0
        # ret_metrics_class.pop('Class', None)
        # for key, value in ret_metrics_class.items():
            # eval_results.update({
                # key + '.' + str(name): value[idx] / 100.0
                # for idx, name in enumerate(class_names)
            # })

        return eval_results
    # def __len__(self):
    #     return len(self.images)
    # def __getitem__(self, idx):
    #     if self.test_mode:
    #         return self.prepare_test_img(idx)
    #     else:
    #         return self.prepare_train_img(idx)
    # def load_annotations(self.img_dir, self.img_suffix, 
    #                     self.ann_dir, 
    #                     self.seg_map_suffix, self.split):
    # def __getitem__(self, index):
    #     image = self.images[index]
    #     label = self.labels[index]
    #     if self.transforms is not None:
    #         image, label = self.transforms(image, label)
    #     return image, label