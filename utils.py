
import os
import glob
import re
import random
import numbers
import queue

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import skimage.morphology as morph


import transforms
from dataset import CamVid, VOC2012Aug, Glas, PreTraining
from conf import settings
from metric import eval_metrics, gland_accuracy_object_level



@torch.no_grad()
def visualize_network(writer, net, tensor):
    tensor = tensor.to(next(net.parameters()).device)
    writer.add_graph(net, tensor)

def _get_lastlayer_params(net):
    """get last trainable layer of a net
    Args:
        network architectur

    Returns:
        last layer weights and last layer bias
    """
    last_layer_weights = None
    last_layer_bias = None
    for name, para in net.named_parameters():
        if 'weight' in name:
            last_layer_weights = para
        if 'bias' in name:
            last_layer_bias = para

    return last_layer_weights, last_layer_bias

def visualize_metric(writer, metrics, values, n_iter):
    for m, v in zip(metrics, values):
        writer.add_scalar('{}'.format(m), v, n_iter)

def visualize_lastlayer(writer, net, n_iter):
    weights, bias = _get_lastlayer_params(net)
    writer.add_scalar('LastLayerGradients/grad_norm2_weights', weights.grad.norm(), n_iter)
    writer.add_scalar('LastLayerGradients/grad_norm2_bias', bias.grad.norm(), n_iter)

def visualize_scalar(writer, name, scalar, n_iter):
    """visualize scalar"""
    writer.add_scalar(name, scalar, n_iter)

def visualize_param_hist(writer, net, n_iter):
    """visualize histogram of params"""
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        #writer.add_histogram("{}/{}".format(layer, attr), param.detach().cpu().numpy(), n_iter)
        writer.add_histogram("{}/{}".format(layer, attr), param, n_iter)

def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it
    Args:
        dataset: instance of torch.nn.Dataset
    Returns:
        return: mean and std of this dataset
    """

    mean = 0
    std = 0

    for img, _ in dataset:
        mean += np.mean(img, axis=(0, 1))

    mean /= len(dataset)

    diff = 0
    for img, _ in dataset:

        diff += np.sum(np.power(img - mean, 2), axis=(0, 1))

    N = len(dataset) * np.prod(img.shape[:2])
    std = np.sqrt(diff / N)

    mean = mean / 255
    std = std / 255

    return mean, std
#def compute_mean_and_std(dataset):
#    """Compute dataset mean and std, and normalize it
#    Args:
#        dataset: instance of torch.nn.Dataset
#
#    Returns:
#        return: mean and std of this dataset
#    """
#
#    mean_r = 0
#    mean_g = 0
#    mean_b = 0
#
#    #opencv BGR channel
#    for img, _ in dataset:
#        mean_b += np.mean(img[:, :, 0])
#        mean_g += np.mean(img[:, :, 1])
#        mean_r += np.mean(img[:, :, 2])
#
#    mean_b /= len(dataset)
#    mean_g /= len(dataset)
#    mean_r /= len(dataset)
#
#    diff_r = 0
#    diff_g = 0
#    diff_b = 0
#
#    N = 0
#
#    for img, _ in dataset:
#
#        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
#        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
#        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))
#
#        N += np.prod(img[:, :, 0].shape)
#
#    std_b = np.sqrt(diff_b / N)
#    std_g = np.sqrt(diff_g / N)
#    std_r = np.sqrt(diff_r / N)
#
#    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
#    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
#    return mean, std

def get_weight_path(checkpoint_path):
    """return absolute path of the best performance
    weight file path according to file modification
    time.

    Args:
        checkpoint_path: checkpoint folder

    Returns:
        absolute absolute path for weight file
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    weight_files = glob.glob(os.path.join(
        checkpoint_path, '*', '*.pth'), recursive=True)

    best_weights = []
    for weight_file in weight_files:
        m = re.search('.*[0-9]+-best.pth', weight_file)
        if m:
            best_weights.append(m.group(0))

    # you can change this to getctime
    compare_func = lambda w: os.path.getmtime(w)

    best_weight = ''
    if best_weights:
        best_weight = max(best_weights, key=compare_func)

    regular_weights = []
    for weight_file in weight_files:
        m = re.search('.*[0-9]+-regular.pth', weight_file)
        if m:
            regular_weights.append(m.group(0))

    regular_weight = ''
    if regular_weights:
        regular_weight = max(regular_weights, key=compare_func)

    # if we find both -best.pth and -regular.pth
    # return the newest modified one
    if best_weight and regular_weight:
        return max([best_weight, regular_weight], key=compare_func)
    # if we only find -best.pth
    elif best_weight and not regular_weight:
        return best_weight
    # if we only find -regular.pth
    elif not best_weight and regular_weight:
        return regular_weight
    # if we do not found any weight file
    else:
        return ''

def get_model(model_name, input_channels, class_num, args=None):

    if model_name == 'unet':
        from models.unet import UNet
        net = UNet(input_channels, class_num)

    elif model_name == 'segnet':
        from models.segnet import SegNet
        net = SegNet(input_channels, class_num)

    elif model_name == 'deeplabv3plus':
        #from models.deeplabv3plus import deeplabv3plus
        from models.deeplabv3plus_tmp import deeplab as deeplabv3plus
        net = deeplabv3plus(class_num)

    elif model_name == 'transunet':
        from models.networks.vit_seg_modeling import transunet
        #net = transunet(settings.IMAGE_SIZE, class_num)
        net = transunet(settings.IMAGE_SIZE, class_num)
    elif model_name == 'medt':
        from models.lib.models.axialnet import gated
        net = gated(img_size=128, imgchan=3)
    elif model_name == 'hybird':
        from models.axial_attention import unet_axial
        net = unet_axial(class_num, args.branch)

    elif model_name == 'axialunet':
        from models.axial_unet import axialunet
        net = axialunet(settings.IMAGE_SIZE)

    elif model_name == 'transseg':
        from models.one.transformer import transseg
        net = transseg(class_num, segment=not args.pretrain)

    elif model_name == 'fullnet':
        from models.fullnet import fullnet
        net = fullnet(class_num)

    elif model_name == 'tg':
        from models.tri_graph import tg
        net = tg(class_num)

    elif model_name == 'mgl':
        from models.mgl.mglnet import mgl
        net = mgl(class_num)

    elif model_name == 'dgcn':
        from models.dual_gcn import DualSeg_res101
        net = DualSeg_res101(class_num)
    else:
        raise ValueError('network type does not supported')

    print()
    print(net)
    print()

    return net

#def intersect_and_union(pred_label, label, num_classes, ignore_index):
#    """Calculate intersection and Union.
#    Args:
#        pred_label (ndarray): Prediction segmentation map
#        label (ndarray): Ground truth segmentation map
#        num_classes (int): Number of categories
#        ignore_index (int): Index that will be ignored in evaluation.
#     Returns:
#         ndarray: The intersection of prediction and ground truth histogram
#             on all classes
#         ndarray: The union of prediction and ground truth histogram on all
#             classes
#         ndarray: The prediction histogram on all classes.
#         ndarray: The ground truth histogram on all classes.
#    """
#
#    mask = (label != ignore_index)
#    pred_label = pred_label[mask]
#    label = label[mask]
#
#    intersect = pred_label[pred_label == label]
#    area_intersect, _ = np.histogram(
#        intersect, bins=np.arange(num_classes + 1))
#    area_pred_label, _ = np.histogram(
#        pred_label, bins=np.arange(num_classes + 1))
#    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
#    area_union = area_pred_label + area_label - area_intersect
#
#    return area_intersect, area_union, area_pred_label, area_label
#
#
#def mean_iou(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None):
#    """Calculate Intersection and Union (IoU)
#    Args:
#        results (list[ndarray]): List of prediction segmentation maps
#        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
#        num_classes (int): Number of categories
#        ignore_index (int): Index that will be ignored in evaluation.
#        nan_to_num (int, optional): If specified, NaN values will be replaced
#            by the numbers defined by the user. Default: None.
#     Returns:
#         float: Overall accuracy on all images.
#         ndarray: Per category accuracy, shape (num_classes, )
#         ndarray: Per category IoU, shape (num_classes, )
#    """
#
#    num_imgs = len(results)
#    assert len(gt_seg_maps) == num_imgs
#    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
#    total_area_union = np.zeros((num_classes, ), dtype=np.float)
#    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
#    total_area_label = np.zeros((num_classes, ), dtype=np.float)
#    for i in range(num_imgs):
#        area_intersect, area_union, area_pred_label, area_label = \
#            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
#                                ignore_index=ignore_index)
#        total_area_intersect += area_intersect
#        total_area_union += area_union
#        total_area_pred_label += area_pred_label
#        total_area_label += area_label
#    all_acc = total_area_intersect.sum() / (total_area_label.sum() + 1e-8)
#    acc = total_area_intersect / (total_area_label + 1e-8)
#    iou = total_area_intersect / (total_area_union + 1e-8)
#    if nan_to_num is not None:
#        return all_acc, np.nan_to_num(acc, nan=nan_to_num), \
#            np.nan_to_num(iou, nan=nan_to_num)
#    return all_acc, acc, iou

def plot_dataset(dataset, out_dir, class_num, num=9, class_id=4, ignore_idx=255):
    colors = [
        [1, 122, 33],
  	    (255,255,255),
 		(255,0,0),
 		(0,255,0),
 		(0,0,255),
 		(255,255,0),
 		(0,255,255),
        (255,0,255),
        (192,192,192),
        (128,128,128),
        (128,0,0),
        (128,128,0),
        (0,128,0),
        (128,0,128),
        (0,128,128),
        (0,0,128),
        (128,0,0),
        (255,255,224),
        (250,250,210),
        (139,69,19),
        (160,82,45),
        (210,105,30),
        (244,164,96),
        (176,196,222),
        (240,255,240),
        (105,105,105),
    ]
    colors = random.choices(colors, k=class_num)
    transforms = dataset.transforms
    for idx in range(num):
        i = random.choice(range(len(dataset)))
        img, label = dataset[i]

        label = label
        ignore_mask = label == ignore_idx
        #print(ignore_idx)
        #if 0 != class_id:
        #    label[ignore_mask] = 0
        #else:
        #    label[ignore_mask] = 1

        #label[label == class_id] = 255
        #label[label != 255] = 0
        out_mask = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
        for cid in range(class_num):
            out_mask[label == cid] = colors[cid]

        out_mask[ignore_mask] = (0, 0, 0)
        img = np.concatenate((img, out_mask), axis=1)
        cv2.imwrite(os.path.join(out_dir, 'label{}.png'.format(idx)), img)



    dataset.transforms = transforms

def print_eval(class_names, results):
    assert len(class_names) == len(results)
    msg = []
    for cls_idx, (name, res) in enumerate(zip(class_names, results)):
        msg.append('{}(id {}): {:.4f}'.format(name, cls_idx, res))

    #msg = msg.strip()
    #msg = msg[:-1]
    msg = 'Total {} classes '.format(len(results)) + ', '.join(msg)

    print(msg)

def pretrain_training_transforms():
    import transforms_pretrain
    trans = transforms_pretrain.Compose([
            #transforms_pretrain.EncodingLable(),
            transforms_pretrain.RandomHorizontalFlip(),
            transforms_pretrain.RandomVerticalFlip(),
            transforms_pretrain.RandomRotation(15, fill=0),
            transforms_pretrain.ColorJitter(0.4, 0.4),
            transforms_pretrain.RandomGaussianBlur(),
            transforms_pretrain.RandomScaleCrop(settings.IMAGE_SIZE),
            transforms_pretrain.ToTensor(),
            transforms_pretrain.Normalize(settings.MEAN, settings.STD),
        ])

    return trans

def pretrain_test_transforms():
    import transforms_pretrain
    trans = transforms_pretrain.Compose([
            #transforms_pretrain.EncodingLable(),
            #transforms_pretrain.RandomHorizontalFlip(),
            #transforms_pretrain.RandomVerticalFlip(),
            #transforms_pretrain.RandomRotation(15, fill=0),
            #transforms_pretrain.ColorJitter(0.4, 0.4),
            #transforms_pretrain.RandomGaussianBlur(),
            transforms_pretrain.RandomScaleCrop(settings.IMAGE_SIZE),
            transforms_pretrain.ToTensor(),
            transforms_pretrain.Normalize(settings.MEAN, settings.STD),
        ])

    return trans

def data_loader(args, image_set):

    if args.pretrain:

        dataset = PreTraining(
                'data/pre_training/',
                image_set=image_set
            )

        if image_set == 'train':
            trans = pretrain_training_transforms()
        if image_set == 'val':
            trans = pretrain_test_transforms()

        dataset.transforms = trans
        data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.b, num_workers=4, shuffle=False, pin_memory=True)

        return data_loader

    if args.dataset == 'Camvid':
        dataset = CamVid(
            'data',
            image_set=image_set,
            download=args.download
        )
    elif args.dataset == 'Glas':
        dataset = Glas(
            'data',
            image_set=image_set,
            download=args.download
        )

    elif args.dataset == 'Voc2012':
        dataset = VOC2012Aug(
            'voc_aug',
            image_set=image_set
        )

    else:
        raise ValueError('datset {} not supported'.format(args.dataset))

    if image_set == 'train':
        #img_scale = (522, 775)
        img_norm_cfg = dict(
        # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
            mean=[200.2103937666394, 130.35073328696086, 200.7955948978498],
            std=[33.296002816472495, 62.533182555417845, 42.21131635702842],
            to_rgb=True
        )


#        trans = mmtransform.Compose(
#            [
#                mmtransform.Resize(img_scale=img_scale, ratio_range=(0.5, 2.0)),
#                mmtransform.RandomRotate(prob=1, degree=(0, 90), auto_bound=True),
#                mmtransform.RandomCrop(crop_size),
#                mmtransform.RandomFlip(prob=0.5, direction='horizontal'),
#                mmtransform.RandomFlip(prob=0.5, direction='vertical'),
#                mmtransform.PhotoMetricDistortion(),
#                mmtransform.Normalize(**img_norm_cfg),
#                mmtransform.Pad(size=crop_size, pad_val=0, seg_pad_val=255),
#                mmtransform.DefaultFormatBundle(),
##my_trans.append(RandomCrop(crop_size=crop_size, cat_max_ratio=0.75))
##my_trans.append(RandomFlip(prob=0.5))
##my_trans.append(PhotoMetricDistortion())
#            ]
#        )
        #trans = transforms.Compose([
        #    transforms.EncodingLable(),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.RandomVerticalFlip(),
        #    transforms.RandomRotation(15, fill=0),
        #    transforms.RandomScaleCrop(settings.IMAGE_SIZE),
        #    transforms.RandomGaussianBlur(),
        #    transforms.ColorJitter(0.4, 0.4),
        #    transforms.ToTensor(),
        #    transforms.Normalize(settings.MEAN, settings.STD),
        #])

        crop_size=(480, 480)
        trans = transforms.Compose([
            transforms.ElasticTransform(alpha=10, sigma=3, alpha_affine=30, p=0.5),
            transforms.Resize(range=[0.5, 1.5]),
            # transforms.RandomRotation(degrees=90, expand=False),
            transforms.RandomRotation(degrees=90, expand=True),
            transforms.RandomCrop(crop_size=crop_size, cat_max_ratio=0.75, pad_if_needed=True),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                transforms=[transforms.PhotoMetricDistortion()]
            ),
            transforms.ToTensor(),
            transforms.Normalize(settings.MEAN, settings.STD)
        ])

    elif image_set == 'val':
        #trans = transforms.Compose([
        #    #transforms.RandomScaleCrop(settings.IMAGE_SIZE),
        #    transforms.EncodingLable(),
        #    transforms.CenterCrop(settings.IMAGE_SIZE, fill=dataset.ignore_index),
        #    transforms.ToTensor(),
        #    transforms.Normalize(settings.MEAN, settings.STD),
        #])

        trans = transforms.MultiScaleFlipAug(
            img_ratios=[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0],
            #flip=True,
            flip=False,
            # flip_direction=['horizontal', 'vertical'],
            flip_direction=['horizontal'],
            # transforms=[
                # transforms.ToTensor(),
                # transforms.Normalize(settings.MEAN, settings.STD),
            # ]
            mean=settings.MEAN,
            std=settings.STD
        )

    elif image_set in ['test', 'testA', 'testB']:
        # trans = transforms.Compose([
        #     transforms.ToTensor(),
        #     #transforms.Normalize(settings.MEAN, settings.STD),
        # ])
        trans = transforms.MultiScaleFlipAug(
            img_ratios=[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0],
            flip=True,
            # flip_direction=['horizontal', 'vertical'],
            flip_direction=['horizontal'],
            transforms=[
                transforms.ToTensor(),
                transforms.Normalize(settings.MEAN, settings.STD),
            ]
        )

    else:
        raise ValueError('image_set should be one of "train", "val", \
                instead got "{}"'.format(image_set))

    print('transforms:')
    print(trans)
    print()
    dataset.transforms = trans


    def multiscale_collate(batch):

        res = []
        for img_meta in batch:
            #res.append([img, gt_seg])
            res.append(img_meta)

        return res

    if image_set == 'test':
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True, persistent_workers=True)
    elif image_set == 'val':
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, num_workers=4, shuffle=False, pin_memory=True, persistent_workers=True,
            collate_fn=multiscale_collate)
    else:
        data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.b, num_workers=4, shuffle=True, pin_memory=True, persistent_workers=True,
                prefetch_factor=4
                )

    return data_loader


def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    input = input.cuda()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0)
    if flip:
        input = torch.cat([input, input.flip(3)], 0)

    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        # output[1] only has 3 dimension
        # 2rd dimension of output[1] equals to
        # third dimension of input
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output

def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    #print(stride_h, stride_w)
    # how many grids vertically, horizontal
    #print('new_h', new_h, 'crop_h', crop_h)
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    #print('new_w', new_w, 'crop_w', crop_w)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)

            # if remaining length is smaller than crop_h
            # align the last crop_h with image border
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]

    # resize to original image size
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

def assign_colors(img, num):
    colors = [
        [1, 122, 33],
  	    # (255,25,255),
 		(255,0,0),
 		(0,255,0),
 		(0,0,255),
 		(255,255,0),
 		(0,255,255),
        (255,0,255),
        (192,192,192),
        (128,128,128),
        (128,0,0),
        (128,128,0),
        (0,128,0),
        (128,0,128),
        (0,128,128),
        (0,0,128),
        (128,0,0),
        (255,255,224),
        (250,250,210),
        (139,69,19),
        (160,82,45),
        (210,105,30),
        (244,164,96),
        (176,196,222),
        (240,255,240),
        (105,105,105),
        (46,139,87),
        (0,0,139),
        (139,0,139),
        (238,130,238),
        (255,250,205),
        (160,82,45),
        (245,255,250),
        (255,228,181),
        (255,245,238),
        (119,136,153),
        (255,105,180),
    ]
    gt_colors = cv2.cvtColor(np.zeros(img.shape).astype('uint8'), cv2.COLOR_GRAY2BGR)

    for i in range(num):
        i += 1
        #print(img.shape, gt_colors.shape)
        #print(np.unique(img), i)
        gt_colors[img == i] = colors[i]

    return gt_colors

def test(net, test_dataloader, crop_size, scales, base_size, classes, mean, std, checkpoint):
    net.eval()


    f1 = 0
    dice = 0
    hausdorff = 0
    recall = 0
    precision = 0
    iou = 0

    image_set = test_dataloader.dataset.image_set
    save_path = os.path.join(settings.EVAL_PATH, checkpoint, image_set)

    msges = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ig_idx = test_dataloader.dataset.ignore_index
    cls_names = test_dataloader.dataset.class_names
    net = net.cuda()
    for i, (img, label) in enumerate(test_dataloader):
        assert test_dataloader.batch_size == 1
        img = img.cuda()
        label = label.cuda()
        label = label.squeeze(dim=0)
        img = np.squeeze(img.cpu().numpy(), axis=0)
        img = np.transpose(img, (1, 2, 0))
        h, w, _ = img.shape
        prediction = np.zeros((h, w, classes), dtype=float)

        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)

            img_scale = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            crop_h = crop_size
            crop_w = crop_size
            prediction += scale_process(net, img_scale, classes, crop_h, crop_w, h, w, mean, std)
            #print(scale, base_size, new_w, new_h, crop_h, crop_w, h, w)

        prediction /= len(scales)
        preds = np.argmax(prediction, axis=2)
        #print('prediction1', prediction.shape)

        #preds = np.expand_dims(prediction, axis=0)
        #tmp_all_acc, tmp_acc, tmp_iou = eval_metrics(
        #    preds,
        #    label.detach().cpu().numpy(),
        #    len(cls_names),
        #    ignore_index=ig_idx,
        #    metrics='mIoU',
        #    nan_to_num=-1
        #)
        tmp_recall, tmp_precision, tmp_f1, tmp_dice, tmp_iou, tmp_haus = gland_accuracy_object_level(
            preds,
            label.detach().cpu().numpy()
        )
        msg = 'img{}, iou: {:.6f}, f1: {:.6f}, recall: {:.6f}, precision: {:.6f}, dice: {:.6f}, hausdorff: {:.6f}'.format(
            i,
            tmp_iou,
            tmp_f1,
            tmp_recall,
            tmp_precision,
            tmp_dice,
            tmp_haus
        )
        print(msg)
        msges.append(msg)

        preds = morph.remove_small_objects(preds == 1, 100)  # remove small object
        pred_labeled = morph.label(preds, connectivity=2)

        gt_labeled = morph.label(label.detach().cpu().numpy(), connectivity=2)
        gt_labeled = morph.remove_small_objects(gt_labeled, 3)   # remove 1 or 2 pixel noise in the image
        gt_labeled = morph.label(gt_labeled, connectivity=2)

        #pred_colors = assign_colors(pred_labeled, np.max(pred_labeled))
        #gt_colors = assign_colors(gt_labeled, np.max(gt_labeled))


        cv2.imwrite(os.path.join(save_path, '{}_pred{}.png').format(image_set, i), pred_labeled)
        cv2.imwrite(os.path.join(save_path, '{}_gt{}.png').format(image_set, i), gt_labeled)
        cv2.imwrite(os.path.join(save_path, '{}_img{}.png').format(image_set, i), (img * std + mean) * 255)


        #all_acc += tmp_all_acc
        #acc += tmp_acc
        iou += tmp_iou
        recall += tmp_recall
        precision += tmp_precision
        f1 += tmp_f1
        dice += tmp_dice
        hausdorff += tmp_haus

    iou /= len(test_dataloader.dataset)
    f1 /= len(test_dataloader.dataset)
    recall /= len(test_dataloader.dataset)
    precision /= len(test_dataloader.dataset)
    dice /= len(test_dataloader.dataset)
    hausdorff /= len(test_dataloader.dataset)
    msg = 'iou: {:.6f}, f1: {:.6f}, recall: {:.6f}, precision: {:.6f}, dice: {:.6f}, hausdorff: {:.6f}'.format(
        iou,
        f1,
        recall,
        precision,
        dice,
        hausdorff
    )
    print(msg)
    msges.append(msg)

    with open(os.path.join(settings.EVAL_PATH, checkpoint, image_set) + '.txt', 'w') as res_file:
        for msg in msges:
            res_file.write(msg + '\n')

    #all_acc /= len(test_dataloader.dataset)
    #acc /= len(test_dataloader.dataset)
    #iou /= len(test_dataloader.dataset)
    #print('Iou for each class:')
    #print_eval(cls_names, iou)
    #print('Acc for each class:')
    #print_eval(cls_names, acc)
    #print('%, '.join([':'.join([str(n), str(round(i, 2))]) for n, i in zip(cls_names, iou)]))
    #iou = iou.tolist()
    #iou = [i for i in iou if iou.index(i) != ig_idx]
    #miou = sum(iou) / len(iou)
    #macc = sum(acc) / len(acc)
    #print('Mean acc {:.4f} Mean iou {:.4f}  All Pixel Acc {:.4f}'.format(macc, miou, all_acc))
    #print('%, '.join([':'.join([str(n), str(round(a, 2))]) for n, a in zip(cls_names, acc)]))
    #print('All acc {:.2f}%'.format(all_acc))



class CheckPointManager:
    def __init__(self, save_path, max_keep_ckpts=5):

        self.lg = [
            'testA_F1',
            'testA_Dice',
            'testB_F1',
            'testB_Dice',
            'total_F1',
            'total_Dice',
        ]

        self.le = [
            'testA_Haus',
            'testB_Haus',
            'total_Haus',
        ]

        self.save_path = save_path
        self.max_keep_ckpts = max_keep_ckpts
        self.ckpt_keep_queue = queue.Queue()

        self.best_value = {}
        self.best_path = {}


    def assert_metric(self, metrics):
        for m in metrics:
            if m not in self.lg and m not in self.le:
                raise ValueError('{} should be in {} or {}'.format(
                    m,
                    self.lg,
                    self.le
                ))

    def assert_values(self, values):
        for v in values:
            assert isinstance(v, numbers.Number)


    def save_ckp_iter(self, model, iter_idx):
        ckpt_save_path = os.path.join(
            self.save_path,
            'iter_{}.pt'.format(iter_idx)
            )

        print('saving checkpoint file to {}'.format(ckpt_save_path))
        torch.save(model.state_dict(), ckpt_save_path)

        self.ckpt_keep_queue.put(ckpt_save_path)

        if self.ckpt_keep_queue.qsize() > self.max_keep_ckpts:
            del_path = self.ckpt_keep_queue.get()
            print('deleting checkpoint file {}'.format(del_path))
            os.remove(del_path)


    def if_update(self, m, v):
        if not isinstance(v, numbers.Number):
            raise ValueError('{} should be a number'.format(v))

        if m in self.lg:
            # the more the better
            if self.best_value[m] < v:
                return True

        if m in self.le:
            # the less the better
            if self.best_value[m] > v:
                return True

        return False


    def save_best(self, model, metrics, values, iter_idx):
        """only keep one ckt for each metric"""
        self.assert_metric(metrics)
        self.assert_values(values)

        for m, v in zip(metrics, values):

            if m not in self.best_value:
                self.best_value[m] = v


            if self.if_update(m, v):

                # saving best ckpt
                ckpt_path = os.path.join(self.save_path,
                            'best_{}_{}_iter_{}.pt'.format(m, v, iter_idx)
                    )
                print('saving best checkpoint file to {}'.format(ckpt_path))
                torch.save(model.state_dict(), ckpt_path)


                # del former best ckpt
                # if former best chpt exists
                if m in self.best_path:
                    print('deleting best checkpoint file {}'.format(self.best_path[m]))
                    os.remove(self.best_path[m])


                # update values
                self.best_value[m] = v
                self.best_path[m] = ckpt_path
