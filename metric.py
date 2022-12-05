import numpy as np
import skimage.morphology as morph
from skimage import measure
from scipy.spatial.distance import directed_hausdorff as hausdorff
import cv2


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.
    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index):
    """Calculate Total Intersection and Union.
    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index=ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None):
    """Calculate Mean Intersection and Union (mIoU)
    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None):
    """Calculate Mean Dice (mDice)
    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category dice, shape (num_classes, )
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category evalution metrics, shape (num_classes, )
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes,
                                                     ignore_index=ignore_index)
    all_acc = total_area_intersect.sum() / (total_area_label.sum() + 1e-8)
    acc = total_area_intersect / (total_area_label + 1e-8)
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / (total_area_union + 1e-8)
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics


import skimage.morphology as morph

#def assign_colors(img, num):
#    colors = [
#        [1, 122, 33],
#  	    (255,255,255),
# 		(255,0,0),
# 		(0,255,0),
# 		(0,0,255),
# 		(255,255,0),
# 		(0,255,255),
#        (255,0,255),
#        (192,192,192),
#        (128,128,128),
#        (128,0,0),
#        (128,128,0)
#    ]
#    gt_colors = cv2.cvtColor(np.zeros(img.shape).astype('uint8'), cv2.COLOR_GRAY2BGR)
#
#    for i in range(num):
#        i += 1
#        gt_colors[img == i] = colors[i]
#
#    return gt_colors



def gland_accuracy_object_level(pred, gt):
    """ Compute the object-level hausdorff distance between predicted  and
    groundtruth """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    #print('gland acc function', pred.shape, gt.shape)
    # remove small object size
    pred = pred == 1
    pred = morph.remove_small_objects(pred, 100)  # remove small object
    #pred = measure.label(pred)
    #pred = morph.dilation(pred, selem=morph.selem.disk(4))


    # get connected components
    pred_labeled = morph.label(pred, connectivity=2)
    pred_labeled = morph.remove_small_objects(pred_labeled)   # remove 1 or 2 pixel noise in the image
    pred_labeled = morph.label(pred_labeled, connectivity=2)

    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = morph.label(gt, connectivity=2)
    gt_labeled = morph.remove_small_objects(gt_labeled)   # remove 1 or 2 pixel noise in the image
    gt_labeled = morph.label(gt_labeled, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1
    #print('ffffff', pred_labeled.shape, gt_labeled.shape)

    #gt_colors = assign_colors(gt_labeled, Ng + 1)
    #pred_colors = assign_colors(pred_labeled, Ns + 1)
    # show_figures((pred_labeled, gt_labeled))

    # --- compute F1 --- #
    TP = 0.0  # true positive
    FP = 0.0  # false positive
    for i in range(1, Ns + 1):
        pred_i = np.where(pred_labeled == i, 1, 0)
        img_and = np.logical_and(gt_labeled, pred_i)

        # get intersection objects in target
        overlap_parts = img_and * gt_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((img_i, overlap_parts))

        # no intersection object
        if obj_no.size == 0:
            FP += 1
            continue

        # find max overlap object
        obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
        gt_obj = obj_no[np.argmax(obj_areas)]  # ground truth object number

        gt_obj_area = np.sum(gt_labeled == gt_obj)  # ground truth object area
        overlap_area = np.sum(overlap_parts == gt_obj)

        if float(overlap_area) / gt_obj_area >= 0.5:
            TP += 1
        else:
            FP += 1

    FN = Ng - TP  # false negative

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt

    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    hausdorff_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        gamma_i = float(np.sum(gt_i)) / gt_objs_area

        # show_figures((pred_labeled, gt_i, overlap_parts))

        if obj_no.size == 0:   # no intersection object
            dice_i = 0
            iou_i = 0

            # find nearest segmented object in hausdorff distance
            min_haus = 1e5
            for j in range(1, Ns + 1):
                pred_j = np.where(pred_labeled == j, 1, 0)
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_i)
                haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                if haus_tmp < min_haus:
                    min_haus = haus_tmp
            haus_i = min_haus
        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

            # compute hausdorff distance
            seg_ind = np.argwhere(pred_i)
            gt_ind = np.argwhere(gt_i)
            haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        hausdorff_g += gamma_i * haus_i


    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

            # find nearest groundtruth object in hausdorff distance
            min_haus = 1e5
            for i in range(1, Ng + 1):
                gt_i = np.where(gt_labeled == i, 1, 0)
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_i)
                haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                if haus_tmp < min_haus:
                    min_haus = haus_tmp
            haus_j = min_haus
        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

            # compute hausdorff distance
            seg_ind = np.argwhere(pred_j)
            gt_ind = np.argwhere(gt_j)
            haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        hausdorff_s += sigma_j * haus_j

    return recall, precision, F1, (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2

#results = np.random.randint(0, 5, size=(2, 3, 3))
#preds = np.random.randint(0, 5, size=(2, 3, 3))
#preds = results
#print(results)
#print("preds", preds)
#
#num_classes = 6
#print(np.histogram(results, bins=num_classes+1))
#
#
#all_acc, acc, iou = eval_metrics(
#        results,
#        preds,
#        num_classes,
#        ignore_index=6,
#        metrics='mIoU',
#        nan_to_num=-1)
#print(round(all_acc, 4), acc, iou)