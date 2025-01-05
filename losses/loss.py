# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_metric_learning import losses


import skimage.morphology as morph
from skimage import measure
import numpy as np
import cv2
import unet3
import time

def connected_components(image: torch.Tensor, num_iterations: int = 100) -> torch.Tensor:
    r"""Computes the Connected-component labelling (CCL) algorithm.
    .. image:: https://github.com/kornia/data/raw/main/cells_segmented.png
    The implementation is an adaptation of the following repository:
    https://gist.github.com/efirdc/5d8bd66859e574c683a504a4690ae8bc
    .. warning::
        This is an experimental API subject to changes and optimization improvements.
    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       connected_components.html>`__.
    Args:
        image: the binarized input image with shape :math:`(*, 1, H, W)`.
          The image must be in floating point with range [0, 1].
        num_iterations: the number of iterations to make the algorithm to converge.
    Return:
        The labels image with the same shape of the input image.
    Example:
        >>> img = torch.rand(2, 1, 4, 5)
        >>> img_labels = connected_components(img, num_iterations=100)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input imagetype is not a torch.Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"Input image shape must be (*,1,H,W). Got: {image.shape}")

    with torch.no_grad():
        H, W = image.shape[-2:]
        image_view = image.view(-1, 1, H, W)

        # precompute a mask with the valid values
        mask = image_view == 1

        # allocate the output tensors for labels
        B, _, _, _ = image_view.shape
        out = torch.arange(B * H * W, device=image.device, dtype=image.dtype).view((-1, 1, H, W))
        out[~mask] = 0

        for _ in range(num_iterations):
            out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

        return out.view_as(image)
#

import time
from functools import wraps

def time_execution(func):
    @wraps(func)
    def wrapper(self, args, kwargs):
        t1 = time.time()
        func(self, *args, **kwargs)
        t2 = time.time()
        print(t2 - t1)
    return wrapper

class GlandContrastLoss(nn.Module):
    def __init__(self, num_nagative, rate=30,temperature=0.07, ignore_idx=255):
        super().__init__()
        print("GlandContrastLoss",num_nagative)
        #self.grid_size = grid_size
        self.num_nagative = num_nagative
        self.op = 'xor'
        self.rate = rate
        print(f'rate:{rate}')
        self.temperature = temperature
        self.base_temperature = 0.1
        #self.infonce_loss_fn = losses.NTXentLoss(temperature=0.07)
        self.store_values = {
        }

        self.ignore_idx = ignore_idx

        self.total_time = 0
        self.total_samples = 0

    # @time_execution
    def segment_level_loss(self, gt, pred, op='xor', out_size=(160, 160)):
        # print('????????')
        # assert out_size[0] == out_size[1]
        
        # h, w = out_size

        # print("gt")
        # print(gt.shape)
        # print("pred")
        # print(pred.shape)
        # exit(1)
        gt = cv2.resize(gt, out_size[::-1], interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred, out_size[::-1], interpolation=cv2.INTER_NEAREST)

        count_over_conn = 0
        count_under_conn = 0


        if op == 'none':
            return np.zeros(gt.shape, dtype=np.uint8)


        # remove ignore_idx
        # pred idx only contains 0 or 1, so we need to remove the blank region acoording = gt
        # print("ignore_idx")
        # print(self.ignore_idx)
        # print(gt.shape)
        # print(pred.shape)
        # print(gt==self.ignore_idx)
        # exit(1)
        # with open('feats_output——xxn_pred_old.txt', 'w') as f:
        #     # 如果 feats 是一个列表或包含较复杂数据结构，你可以遍历并写入
        #     for row in pred:
        #         line = " ".join(map(str, row))
        #         f.write(line + "\n")
        # with open('feats_output——xxn_gt.txt', 'w') as f:
        #     # 如果 feats 是一个列表或包含较复杂数据结构，你可以遍历并写入
        #     for row in gt:
        #         line = " ".join(map(str, row))
        #         f.write(line + "\n")   
        # print(type(gt))
        # exit(1)
        pred[gt==self.ignore_idx] = 0
        # print("pred[gt==self.ignore_idx]")
        # print(pred)
        # with open('feats_output——xxn_pred_new.txt', 'w') as f:
        #     # 如果 feats 是一个列表或包含较复杂数据结构，你可以遍历并写入
        #     for row in pred:
        #         line = " ".join(map(str, row))
        #         f.write(line + "\n")
        # exit(1)



        # set connectivity to 1 and set min_size to 64 as default
        # because some of the groud truth gland will have sawtooth effect due to the data augmentation
        pred = morph.remove_small_objects(pred == 1, connectivity=1)
        gt = morph.remove_small_objects(gt == 1, connectivity=1)



        # set connectivity to 1 to avoid glands clustered together due to resize
        # only counts cross connectivity
        pred_labeled, pred_num = measure.label(pred, return_num=True, connectivity=1)
        gt_labeled, gt_num = measure.label(gt, return_num=True, connectivity=1)
        # print("pred_num")
        # # 第一次40
        # print(pred_num)
        # print("gt_num")
        # # 第一次2
        # print(gt_num)
        # exit(1)
        # with open('feats_output——xxn_pred_labeled.txt', 'w') as f:
        #     # 如果 feats 是一个列表或包含较复杂数据结构，你可以遍历并写入
        #     for row in pred_labeled:
        #         line = " ".join(map(str, row))
        #         f.write(line + "\n")
        # exit(1)
        #gt_colors = assign_colors(gt_labeled, gt_num)
        #pred_colors = assign_colors(pred_labeled, pred_num)
        #cv2.imwrite('gt_colors.jpg', gt_colors)
        #cv2.imwrite('pred_colors.jpg', pred_colors)
        #pred1 = morph.dilation(pred, selem=morph.selem.disk(4))

        #cv2.imwrite('pred_loss.png', pred_labeled / pred_labeled.max() * 255)

        #num_pred_objs = len(np.unique(pred))
        #res = np.zeros(pred_labeled.shape)

        #colors = random.choices(colors, k=pred_num)
        #gt_colors = cv2.cvtColor(np.zeros(gt_labeled.shape).astype('uint8'), cv2.COLOR_GRAY2BGR)
        #pred_colors = gt_colors.copy()
        #g_num = np.unique(gt_labeled)
        #p_num = np.unique(pred_labeled)
        #print('p_num', p_num)
        #for i in g_num:
        #    if i == 0:
        #        continue
        #    gt_colors[gt_labeled == i] = colors[i]
        #cv2.imwrite('gt_colors.png', gt_colors)

        #for i in p_num:
        #    if i == 0:
        #        continue
        #    pred_colors[pred_labeled == i] = colors[i]
        #cv2.imwrite('pred_colors.png', pred_colors)



        # iterate through prediction
        #print(np.unique(gt), np.unique(pred))
        results = []
        #pred
        res = np.zeros(gt.shape, dtype=np.uint8)
        ans = np.zeros(gt.shape, dtype=np.uint8)
        process_list=[]
        rate = self.rate
        
        # based on pred glands
        # print('pred_num', pred_num)
        for i in range(0, pred_num):
            i += 1

            # gt != 0 is gt gland
            # pred_labeled == i is the ith gland of pred
            pred_labeled_i = pred_labeled == i
            # with open('feats_output——xxn_pred_labeled_i.txt', 'w') as f:
            # # 如果 feats 是一个列表或包含较复杂数据结构，你可以遍历并写入
            #     for row in pred_labeled_i:
            #         line = " ".join(map(str, row))
            #         f.write(line + "\n")
                    
            # exit(1)
            #返回的也是一个boolean数组
            # 这里的mask应该是预测的和真实值对上的
            mask = (pred_labeled_i) & (gt != 0)

            # for each pixel of mask in corresponding gt img
            #print(gt_labeled[mask].shape[0], len(gt_labeled[mask]))
            # print(len(gt_labeled[mask]))
            # exit(1)
            if len(gt_labeled[mask]) == 0:
                # no gt gland instance in corresponding
                # location of gt image

                #res[pred_labeled == i] = 1
                res[pred_labeled_i] = 1
                ans[pred_labeled_i] = 1
                count_over_conn += 1

                continue

            # one pred gland contains more than one gt glands
            if gt_labeled[mask].min() != gt_labeled[mask].max():
                # more than 1 gt gland instances in corresponding
                # gt image
                #res[pred_labeled == i] = 1
                res[pred_labeled_i] = 1
                
                # 处理gt
                gt_unique_values=np.unique(gt_labeled[mask])
                new_pic = np.zeros(gt.shape, dtype=np.uint8)
                
                for value in gt_unique_values:
                    new_pic[gt_labeled == value] = 1
                    
                w = unet3.weight_add_np(new_pic, rate)
                # 是否正则？
                w = w / w.max() 
                process_list.append(w)
                
                count_over_conn += 1

            else:
                # corresponding gt gland area is less than 50%
                if mask.sum() / pred_labeled_i.sum() < 0.5:
                    #res[pred_labeled == i] = 1
                    res[pred_labeled_i] = 1
                    ans[pred_labeled_i] = 1
                    count_over_conn += 1
                    #pred_labeled_i_xor = np.logical_xor(pred_labeled_i, mask)
                    #res[pred_labeled_i_xor] = 1


        # vis
        ###################################################
        # cv2.imwrite('my_mutal_alg/pred_region_wrong_number.png', res * 255)
        ###################################################

        #gt
        results.append(res)

        res = np.zeros(gt.shape, dtype=np.uint8)
        # print('gt_num', gt_num)
        for i in range(0, gt_num):
            i += 1
            gt_labeled_i = gt_labeled == i
            #mask = (gt_labeled == i) & (pred != 0)
            mask = gt_labeled_i & (pred != 0)

            if len(pred_labeled[mask]) == 0:
                # no pred gland instance in corresponding
                # predicted image

                #res[gt_labeled == i] = 1
                res[gt_labeled_i] = 1
                ans[gt_labeled_i] = 1
                #cv2.imwrite('resG{}.png'.format(i), res)
                count_under_conn += 1
                continue

            if pred_labeled[mask].min() != pred_labeled[mask].max():
                #res[gt_labeled == i] = 1
                res[gt_labeled_i] = 1
                #cv2.imwrite('resG{}.png'.format(i), res)
                count_under_conn += 1
                # 处理pred
                pred_unique_values=np.unique(pred_labeled[mask])
                new_pic = np.zeros(pred.shape, dtype=np.uint8)
                
                for value in pred_unique_values:
                    new_pic[pred_labeled == value] = 1
                    
                w = unet3.weight_add_np(new_pic, rate)
                # print("test")
                # print(w.max())
                # print(np.isnan(w.max()))
                w = w / w.max() 
                process_list.append(w)

            else:
                if mask.sum() / gt_labeled_i.sum() < 0.5:
                    #print(mask.sum() / gt_labeled_i.sum(), 'ccccccccccc')
                    #print(i, i, i, i)
                    #res[gt_labeled == i] = 1
                    res[gt_labeled_i] = 1
                    ans[gt_labeled_i] = 1
                    count_under_conn += 1
            #print(mask.sum() / (gt_labeled == i).sum(), 'cc111')
            #start = time.time()
            #for i in range(100):
            #    np.unique(test)
            #finish = time.time()
            #print('unique', finish - start)

            #start = time.time()
            #for i in range(100):
            #    if len(test) == 0:
            #        print('no')
            #        break
            #    test.max()

            #finish = time.time()
            #print('max min', finish - start)

        # vis
        ###################################################
        # cv2.imwrite('my_mutal_alg/gt_region_wrong_number.png', res * 255)
        ###################################################
        results.append(res)


        res = cv2.bitwise_or(results[0], results[1])

        # vis
        ###################################################
        # cv2.imwrite('my_mutal_alg/merge_all_wrong_number_region.png', res * 255)
        ###################################################
        if op == 'or':
            return res

        elif op == 'xor':

            #cc = res.copy()
            gt_res = np.zeros(gt.shape, dtype=np.uint8)
            for i in range(0, gt_num):
                i += 1
                if res[gt_labeled == i].max() != 0:
                    gt_res[gt_labeled == i] = 1

            # vis
            ###################################################
            # cv2.imwrite('my_mutal_alg/gt_1_final_candidate_region.png', gt_res * 255)
            ###################################################
            pred_res = np.zeros(gt.shape, dtype=np.uint8)
            for i in range(0, pred_num):
                i += 1
                if res[pred_labeled == i].max() != 0:
                    pred_res[pred_labeled == i] = 1

            # vis
            ###################################################
            # cv2.imwrite('my_mutal_alg/pred_1_final_candidate_region.png', pred_res * 255)
            ###################################################

            #print(pred_res.shape, 'pred_res.shape')
            res = cv2.bitwise_xor(pred_res, gt_res)
            for index, pic in enumerate(process_list):
                
                pic_xor = pic * res
                # pic_xor=res
                # pic_xor = pic & res
                ans[pic_xor > 0.3]=1
                # ans[pic_xor != 0]=1
            # vis
            ###################################################
            # cv2.imwrite('my_mutal_alg/final_result.png', res * 255)
            ###################################################
            #print(pred_num)
            # return res
            return ans
            #return pred_res
            #return gt_res, pred_res, res, cc
                #if len(pred_labeled[mask]) == 0:
                #if gt_labeled[]:
                #    # no pred gland instance in corresponding
                #    # predicted image

                #    res[gt_labeled == i] = 1
                #    #cv2.imwrite('resG{}.png'.format(i), res)
                #    continue
                #res = cv2.bitwise_or(pred_res, gt_res)
        else:
            raise ValueError('operation not suportted')

    def segment_mask(self, gts, preds, op='or', out_size=(160, 160)):
        bs = gts.shape[0]
        # print("bs")
        # print(bs)
        # exit(1)
        preds = np.argmax(preds, axis=1)
        # print("preds")
        # print(preds.shape)
        # exit(1)
        #for b_idx in range(batch_size):
        #    res.append(segment_level_loss(gt[b_idx], preds[b_idx]))
        t1 = time.time()
        # (480, 480)
        # print("gt0")
        # print(gts[0].shape)
        # print("pred0")
        # (480, 480)
        # print(preds[0].shape)
        
        res = [self.segment_level_loss(gt=gts[b], pred=preds[b], op=op, out_size=out_size) for b in range(bs)]
        t2 = time.time()
        # print("time")
        # print((t2 - t1) / bs)
        # exit(1)
        # print("res")
        # 长度是8
        # print(len(res))
        # (480, 480)
        # print(res[0].shape)
        
        # exit(1)
        
        self.total_time += (t2 - t1)
        self.total_samples += bs
        # print('avg time:', self.total_time / self.total_samples)
        #import sys; sys.exit()
        res = np.stack(res, axis=0)
        return res

    def _compute_indices(self, mask_tensor, num_samples_keep):
        #assert tensor.dim() == 4
        #population = tensor.permute(0, 2, 3, 1)
        #mask_tensor.
        #print(mask_tensor.shape)
        mask_tensor = mask_tensor.contiguous().view(-1)
        t_indices = mask_tensor.nonzero()
        #print(t_indices.shape, 'ccccccccc')
        num_samples = t_indices.shape[0]
        perm = torch.randperm(num_samples)

        idx = t_indices[perm[:num_samples_keep]]
        #print(idx)

        #return population[idx]

        return idx


    def _sample_feats_even(self, pred_logits, candidate_mask, gt_seg, ignore_mask):

        assert candidate_mask.dim() == 4
        assert gt_seg.shape == candidate_mask.shape

        # masks of  xor_mask


        batch_size = pred_logits.shape[0]
        logits_dim = pred_logits.shape[1]

        # check if there is an image do not contain any object (value == 1 is object)
        # mask shape: [B, 1]
        mask = candidate_mask.sum(dim=(2, 3)) == 0

        # assign the gt gland to the non-gland image
        candidate_mask[mask] = gt_seg[mask].long()

        #if 'candidate_mask_gland' in self.store_values.keys():
        #    self.store_values['candidate_mask_bg'] = [g for g in candidate_mask.clone()]
        #else:
        #    self.store_values['candidate_mask_gland'] = [g for g in candidate_mask.clone()]


        # randomly sample k elements in candidate_mask
        candidate_mask = candidate_mask + torch.rand(candidate_mask.shape, device=candidate_mask.device)
        candidate_mask[ignore_mask] = 0
        #print(candidate_mask.shape)
        #if 'candidate_mask_gland_rand' in self.store_values.keys():
        #    self.store_values['candidate_mask_bg_rand'] = [g for g in candidate_mask.clone()]
        #else:
        #    self.store_values['candidate_mask_gland_rand'] = [g for g in candidate_mask.clone()]

        _, candidate_indices = candidate_mask.view(batch_size, -1).topk(k=self.num_nagative, dim=-1)
        #[8, 1, 122, 122],  [8, 256, 122, 122]
        #print(candidate_mask.shape, pred_logits.shape)


        pred_logits = pred_logits.permute(0, 2, 3, 1).view(batch_size, -1, logits_dim)
        candidate_indices = candidate_indices.unsqueeze(-1).expand(-1, -1, pred_logits.shape[-1])
        out = torch.gather(input=pred_logits, index=candidate_indices, dim=1)

        return out




    def hard_sampling_even(self, pred_logits, gt_seg, xor_mask):
        #sample hard examples for each image in a batch evenly

        # sampling feat from a 4D tensor
        assert pred_logits.dim() == 4
        batch_size, dim, h, w = pred_logits.shape
        #h = 480
        #w = 480

        labels = gt_seg.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (h, w), mode='nearest').long()
        #labels = gt_seg.unsqueeze(1)

        xor_mask = xor_mask.unsqueeze(1).float().clone()
        xor_mask = torch.nn.functional.interpolate(xor_mask, (h, w), mode='nearest').long()





        #self.store_values['xor_mask'] = [xo for xo in xor_mask]
        #self.store_values['labels'] = [la for la in labels.clone()]


        gland_gt = labels == 1

        #values = torch.amax(gland_gt, dim=(-1, -2))
        #assert values.sum() == gland_gt.shape[0]


        gland_hard_mask = gland_gt * xor_mask

        #self.store_values['gland_hard_mask'] = [gland for gland in gland_hard_mask.clone()]


        bg_gt = labels == 0

        bg_hard_mask = bg_gt * xor_mask

        #values = torch.amax(bg_hard_mask, dim=(-1, -2))
        #assert values.sum() == bg_hard_mask.shape[0]
        #self.store_values['bg_hard_mask'] = [gland for gland in bg_hard_mask.clone()]

        ignore_mask = labels == self.ignore_idx

        gland_feats = self._sample_feats_even(pred_logits, gland_hard_mask, labels == 1, ignore_mask)
        bg_feats = self._sample_feats_even(pred_logits, bg_hard_mask, labels == 0, ignore_mask)


        return gland_feats, bg_feats

        # if no gland in an image
        #mask_gland = gland_hard_mask.sum(dim=(2, 3)) == 0

        ## assign the gt gland to the non-gland image
        #print(gland_hard_mask.shape, labels.shape)
        #gland_hard_mask[mask_gland] = labels[mask_gland]

        #gland_hard_mask = gland_hard_mask + torch.rand(gland_hard_mask.shape, device=gland_hard_mask)


        #print(mask_gland)
        #print(mask_gland.shape)
        #print(mask_gland == 0)
        #print(gland_hard_mask.shape)
        #gland_feats = []
        #for batch_idx in range(batch_size):

        #    gland_mask = gland_hard_mask[batch_idx]

        #    # sample hard gland pixels:
        #    if gland_hard_mask.sum() == 0:
        #        gland_mask = (gt_seg == 1).long()

        #    gland_indices = self._compute_indices(gland_mask, self.num_nagative)
        #    gland_logits = pred_logits[batch_idx]

        #    if
        #    bg_indices = self._compute_indices(gland_mask, self.num_nagative)



        #    gland_feats.append()





    def hard_sampling(self, pred_logits, gt_seg, xor_mask):
        #batch_size = pred_logits.shape[0]
        #dim = pred_logits.shape[1]
        #print(pred_logits.shape)
        #print(pred_logits.shape)
        #print(gt_seg.shape)
        #pred_logits
        #h = pred_logits.shape[-2]
        #w = pred_logits.shape[-1]
        assert pred_logits.dim() == 4
        batch_size, dim, h, w = pred_logits.shape
        #print(gt_seg.dtype)
        labels = gt_seg.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (h, w), mode='nearest').long()

        xor_mask = xor_mask.unsqueeze(1).float().clone()
        xor_mask = torch.nn.functional.interpolate(xor_mask, (h, w), mode='nearest').long()
        #print(xor_mask.shape)

        gland_hard_mask = labels & xor_mask
        gt_inv = ~labels
        bg_hard_mask = gt_inv & xor_mask

        #gland_indices = gland_hard.nonzero()
        #num_gland = gland_indices.shape[0]
        #perm = torch.randperm(num_gland)

        #bg_indices = gland_hard.nonzero()
        #num_bg = bg_indices.shape[0]
        #print(gland_hard)
        #print(pred_logits.shape, gland_hard.squeeze(1).shape)

        #flattened

        gland_indices = self._compute_indices(gland_hard_mask, self.num_nagative * batch_size)
        bg_indices = self._compute_indices(bg_hard_mask, self.num_nagative * batch_size)
        #print(gland_indices.shape)
        #print(bg_indices.shape)

        population = pred_logits.permute(0, 2, 3, 1)
        population = population.contiguous().view(-1, dim)
        #print(gland_indices[0], population.shape)
        #print(population[].shape)
        #import sys; sys.exit()
        #print(population.shape, gland_indices.shape)
        gland_feats = population[gland_indices]
        bg_feats = population[bg_indices]
        #print(gland_feats.shape, bg_feats.shape)
        #return gland_indices, bg_indices

        #print(gland_indices.shape)




        # samples [B, num_classes, num_samples_per_class, logits_dim]

        return gland_feats, bg_feats


    # def compute_loss(self, feat, y_feat, queue):
    #     assert feat.shape == pos_feat.shape
    #     assert pos_queue.shape == neg_queue.shape


    #     queue_y = torch.cat([
    #         torch.zeros(queue.shape, device=queue.device, dtype=torch.long),
    #         torch.ones(queue.shape, device=queue.device, dtype=torch.long)
    #     ], dim=0)
    #     print(queue_y)
    #     queue_feat = queue.view(queue.shape[0] * queue.shape[1], queue.shape[2])
    #     #queue_y = torch.


    #     # neg feat
    #     # negative logits: Nx1 #N is the number of pixels sample perclass
    #     neg = torch.einsum('nc,nc->n', [feat, neg_feat]).unsqueeze(-1)

    #     # pos queue
    #     # positive logits: Nxk #k is the length of queue
    #     pos = torch.einsum('nc,ck->nk', [feat, pos_queue.clone().detach()])

    #     # neg queue
    #     # negative logits: Nxk

   # def contrasive(self, gland_feats, bg_feats, queue):

   #     pos_gland = torch.einsum('n')

    def infonce(self, logits_pos, logits_neg):
        numberator = torch.exp(logits_pos / self.temperature)
        denominator = torch.exp(logits_neg / self.temperature).sum(dim=1).unsqueeze(-1) + numberator
        log_exp = -torch.log((numberator / denominator)).sum(dim=1) / logits_pos.shape[-1]

        return log_exp.mean()

    def contrasive_single_class(self, anchor, postive, negative):
        #print(anchor.shape, postive.T.shape)

        #num_pos = postive.shape[0]
        
        # 进行爱因斯坦求和约定运算
        logits_pos = torch.einsum('nc, ck->nk', [anchor, postive.T])
        logits_neg = torch.einsum('nc, ck->nk', [anchor, negative.T])
        # print(logits_pos.shape)
        # torch.Size([32, 1000])
        # exit(1)

        #numberator = torch.exp(logits_pos / self.temperature)
        #denominator = torch.exp(logits_neg / self.temperature).sum(dim=1).unsqueeze(-1) + numberator
        #log_exp = -torch.log((numberator / denominator)).sum(dim=1) / logits_pos.shape[-1]

        #exp_pos = torch.exp(logits_pos / self.temperature)
        #exp_neg = torch.exp(logits_neg / self.temperature)
        #exp_neg_sum = exp_neg.sum(dim=-1)
        #log_exp = -torch.log(exp_pos / (exp_neg_sum + exp_pos)).sum(dim=-1) / logits_pos.shape[-1]

        return self.infonce(logits_pos, logits_neg)


        #return log_exp.mean()








    def constrasive(self, x_feat, labels, queue):
        assert queue.dim() == 3

        num_classes, q_len, dim = queue.shape
        batch_size, num_samples = x_feat.shape[0], x_feat.shape[1]


        assert dim == x_feat.shape[-1]

        assert x_feat.shape[0] == labels.shape[0]



        # 20000, 1
        # q_len * num_classes
        queue_y = torch.cat([
            torch.zeros([1, q_len], device=queue.device, dtype=torch.long),
            torch.ones([1, q_len], device=queue.device, dtype=torch.long)
        ], dim=0).view(-1, 1)


        queue_feat = queue.view(-1, dim) ###########################################

        labels = labels.contiguous().view(-1, 1)
        #print(labels.shape, queue_y.shape) [12, 1] * [10000, 1]
        mask = torch.eq(labels, queue_y.T).float()


        #x_feat = x_feat.view(batch_size * num_samples, dim)

        #print(x_feat.device, queue_feat.device)
        #print(x_feat.shape, queue_feat.T.shape) torch.Size([12, 2, 256]) torch.Size([256, 10000])
        x_feat = torch.nn.functional.normalize(x_feat, dim=1, p=2)
        #print(x_feat.shape)
        #print((x_feat[0] * x_feat[0]).sum())
        similirity_feat_queue = torch.div(torch.matmul(x_feat, queue_feat.T),
                                        self.temperature)

        #print(similirity_feat_queue.max())
        # torch.Size([12, 2, 10000])
        # print(similirity_feat_queue.shape)
        # print(queue_y.shape, feat_queue.shape)
        logits_max, _ =  torch.max(similirity_feat_queue, dim=1, keepdim=True)

        logits = (similirity_feat_queue - logits_max.detach())

        neg_mask = 1 - mask
        #print(neg_maamk)

        #gland_feats = gland_feats.view(-1, gland_feats.shape[-1])
        #bg_feats = bg_feats.view(-1, bg_feats.shape[-1])

        #pos =

        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(batch_size).view(-1, 1).cuda(),
        #                                              0)
        #self.compute_loss(gland_feats, )
        #print(logits_mask.shape)
        #mask = mask * logits_mask

        exp_logits = torch.exp(logits) # [num_samples, q_len]
        pos_logits = exp_logits * mask + 1e-7

        # sum of all negative samples
        neg_logits = exp_logits * neg_mask + 1e-7
        #print(neg_logits.shape) [num_samples, q_len]
        neg_logits = neg_logits.sum(1, keepdim=True)

        log_prob = torch.log(pos_logits) - torch.log(pos_logits + neg_logits)
        #print(log_prob)
        #print(log_prob.shape)


        mean_log_prob_pos = log_prob.sum(1) / mask.sum(1)
        #print(mean_log_prob_pos.shape)

        loss = - mean_log_prob_pos.mean()
        #loss = loss.mean()
        #print('loss', loss)

        return loss






        # 0 galnd 1 bg
#        anchor_feats = torch.stack(
#            [
#                gland_feats.view(-1, gland_feats.shape[-1]),
#                bg_feats.view(-1, bg_feats.shape[-1])
#            ], dim=0)
#
        #pos = torch.einsum('nc, nc->n')

        #pos = anchor_feats

    def _dequeue_and_enqueue(self, gland_feats, bg_feats, queue, queue_ptr):
        #batch_size, num_samples, dim = gland_feats.shape
        num_feats, dim = gland_feats.shape
        num_classes, q_len, dim = queue.shape

        ptr = queue_ptr[0].long().item()

        feats = torch.stack(
            [
                #gland_feats.view(-1, dim),
                #bg_feats.view(-1, dim)
                gland_feats,
                bg_feats
            ],
            dim=0
        )

        #feats = torch.nn.functional.normalize(feats, dim=2, p=2)
        #print(feats[1, 3, :30])

        start = ptr
        end = ptr + num_feats
        #print(start, end)

        #print(feats.shape, queue.shape, start, end)
        if end  > q_len - 1:
            queue[:, - num_feats:, :] = feats
            queue_ptr[0] = 0

        else:
            queue[:, start : end, :] = feats
            queue_ptr[0] = end

    #def my_infonce(self, gland, labels):

    def compute_loss(self, feats, queue):

        embedings = torch.cat([
            feats[:1, :],
            queue,
        ])

        labels = torch.arange(
        )
        labels[:feats.shape[0]] = 0
        print(embedings.shape, labels.shape)

        return self.infonce_loss_fn(embedings, labels)

    def _save_img(self, t, save_path):
        #print(gt_seg.shape)

        for i in enuermt:
            #print(i.shape)

            img_name = os.path.join(save_path, '{}.jpg')
            print(img_name)

            i.permute(1, 2, 0)


    def cal_uncertain_mask(self, pred_logits, mask):
        # print(pred_logits.shape)
        # exit(1)
        pred_probs = pred_logits.softmax(dim=1)
        # print(pred_probs.shape)
        # exit(1)
        diff = torch.abs(pred_probs[:, 0, :, :] - pred_probs[:, 1, :, :])
        # print(diff.shape)
        # exit(1)
        out = diff
        #out = torch.exp(- 5 * diff ** 2)
        #out = torch.exp(diff ** 2 / 0.7)
        #out = torch.exp(-5 * diff ** 2 )
        # # torch.Size([8, 480, 480])
        # print(mask.shape)
        out = out * mask
        # # torch.Size([8, 480, 480])
        # print(out.shape)
        # exit(1)
        return out


    def compute_sample_weight(self, gt_seg, uncertain_mask, class_id):

        gt_mask = gt_seg == class_id
        object_hard_mask = gt_mask * uncertain_mask

        #object_mask = object_hard_mask
        weight_mask = object_hard_mask.sum(dim=(-1, -2)) == 0
        # torch.Size([8, 480, 480])
        # print(gt_mask.shape)
        # # torch.Size([8])
        # print(weight_mask.shape)
        # exit(1)
        #print(object_hard_mask)
        object_hard_mask[weight_mask] = gt_mask[weight_mask].float()

        ignore_mask = gt_seg == self.ignore_idx

        object_hard_mask[ignore_mask] = 0

        return object_hard_mask


    def indices_to_points(self, indices, H, W):
        #x =
        #print(indices.dtype)
        row = torch.div(indices, H, rounding_mode='trunc') / (H - 1)
        # print(row)

        col = indices % W / (W - 1)
        # print(col.shape)
        # exit(1)
        #print('indices', indices)
        #print('row', row)
        #print('col', col)
        # torch.Size([8, 4, 2])
        xy = torch.stack([col, row], dim=-1)
        # print(xy.shape)
        # exit(1)
        #print(xy * 8)

        return xy

    def denormalize(self, grid):
        """Denormalize input grid from range [0, 1] to [-1, 1]

        Args:
            grid (torch.Tensor): The grid to be denormalize, range [0, 1].

        Returns:
            torch.Tensor: Denormalized grid, range [-1, 1].
        """

        return grid * 2.0 - 1.0

    def point_sample(self, feats, points, align_corners):

        add_dim = False
        #print(feats.shape, points.shape, 'befiore')
        if points.dim() == 3:
            add_dim = True
            points = points.unsqueeze(2)
            output = F.grid_sample(
                feats, self.denormalize(points), align_corners=align_corners)
        if add_dim:
            output = output.squeeze(3)
        #print(feats.shape, points.shape, output.shape, 'after')

        output = output.permute(0, 2, 1)
        return output

    def get_points_train(self, gt_seg, uncertain_mask):
        batch_size, H, W = gt_seg.shape

        assert gt_seg.shape == uncertain_mask.shape

        # cal sample weixght
        gland_weight = self.compute_sample_weight(gt_seg, uncertain_mask, class_id=1)
        bg_weight = self.compute_sample_weight(gt_seg, uncertain_mask, class_id=0)
        # torch.Size([8, 480, 480])
        # torch.Size([8, 230400])     
        # 4
        # print(gland_weight.shape)
        # print(gland_weight.view(batch_size, -1).shape)
        # print(self.num_nagative)
        # exit(1)
        # sample accordint to weights
        gland_indices = torch.multinomial(gland_weight.view(batch_size, -1), self.num_nagative, replacement=False)
        try:
           bg_indices = torch.multinomial(bg_weight.view(batch_size, -1), self.num_nagative, replacement=False)
        except:
            print('except...........')
            bg_indices = torch.zeros(batch_size, self.num_nagative, device=gland_indices.device)
        # torch.Size([8, 4])
        # print(bg_indices.shape)
        # tensor([[122252, 148601, 122192,  30538],
        # [104849,  68165, 228229, 113965],
        # [ 10732, 121119, 137542, 104934],
        # [ 54005, 180010, 223157,  69183],
        # [ 91597, 168126, 196380, 225184],
        # [ 99247, 157195,  73606, 185858],
        # [124209, 142814, 132684, 217508],
        # [226490, 120620, 119223, 222603]], device='cuda:0')
        # print(gland_indices)
        # exit(1)
        # convert indices to points
        gland_points = self.indices_to_points(gland_indices, H, W)
        bg_indices = self.indices_to_points(bg_indices, H, W)

        return gland_points, bg_indices


        # convert gland_indices to point_coords [0, 1]


        # gland_gt = labels == 1
        # gland_hard_mask = gland_gt * uncertain_mask

        # bg_gt = labels == 0
        # bg_hard_mask = bg_gt * uncertain_mask


        # mask = candidate_mask.sum(dim=(2, 3)) == 0

        # assign the gt gland to the non-gland image
        # candidate_mask[mask] = gt_seg[mask].long()

        #if 'candidate_mask_gland' in self.store_values.keys():
        #    self.store_values['candidate_mask_bg'] = [g for g in candidate_mask.clone()]
        #else:
        #    self.store_values['candidate_mask_gland'] = [g for g in candidate_mask.clone()]


        # randomly sample k elements in candidate_mask
        # candidate_mask = candidate_mask + torch.rand(candidate_mask.shape, device=candidate_mask.device)
        # candidate_mask[ignore_mask] = 0


    def multi_level_point_sample(self, feats, points, align_corners, fcs):
        # <class 'dict'>
        # print(type(feats))
        # BasicLinear(
        # (fc): Sequential(
        #     (0): Linear(in_features=64, out_features=64, bias=True)
        # )
        # )
        # print(type(fcs[0]))
        # print(fcs[0])
        # print(fcs[1])
        # print(fcs[2])
        # print(fcs[3])
        # exit(1)
        out = 0
        #for _, values in feats.items():

        out += fcs[0](self.point_sample(feats['low_level'], points, align_corners))
        # print(out.shape)
        out += fcs[1](self.point_sample(feats['layer2'], points, align_corners))
        # print(out.shape)
        out += fcs[2](self.point_sample(feats['aux'], points, align_corners))
        # print(out.shape)
        out += fcs[3](self.point_sample(feats['out'], points, align_corners))
        #out += fcs[4](self.point_sample(feats['gland'], points, align_corners))

        # print(type(out))
        # print(out.shape)
        # torch.Size([8, 4, 64])
        # print()
        # exit(1)
        return out



    def forward(self, feats, pred_logits, gt_seg, queue=None, queue_ptr=None, fcs=None):


        self.store_values = {}
        # print("queue")
        # print(queue.shape)
        # print("fcs")
        # print(fcs)
        # exit(1)
        # print("pred_logits")
        # print("low_level shape:",pred_logits['low_level'].shape)
        # print("layer2 shape:",pred_logits['layer2'].shape)
        # print("aux shape:",pred_logits['aux'].shape)
        # print("out:",pred_logits['out'].shape)
        # print(pred_logits)
        # print(pred_logits.shape)
        # with open('feats_output——xxn_pred_logits.txt', 'w') as f:
        #     # 如果 feats 是一个列表或包含较复杂数据结构，你可以遍历并写入
        #     for key, value in pred_logits.items():
        #         f.write(f"{key}: {value}\n")
        # exit(1)
        with torch.no_grad():
            # 会进来
            # print("进来了")
            # exit(1)
            # print()
            # print("gt_seg.detach().cpu().numpy()的形状:")
            # print(gt_seg.detach().cpu().numpy().shape)
            # print("pred_logits.detach().cpu().numpy()的形状:")
            # print(pred_logits.detach().cpu().numpy().shape)
            
            # print("out_size:")
            # print(gt_seg.shape[-2:])
            # print("op是")
            
            # print(self.op)
            # exit(1)
            mask = self.segment_mask(gts=gt_seg.detach().cpu().numpy(),                                
                                     preds=pred_logits.detach().cpu().numpy(), 
                                     op=self.op, out_size=gt_seg.shape[-2:])
            mask = torch.tensor(mask, dtype=gt_seg.dtype, device=gt_seg.device)
            # print("mask.shape")
            # print(mask.shape)
            # print(gt_seg.dtype)
            # print(gt_seg.device)
            # exit(1)
            # with open('feats_output——xxn_mask.txt', 'w') as f:
            #     for row in mask[0]:
            #         line = " ".join(map(str, row))
            #         f.write(line + "\n")
            # exit(1)
            self.store_values['mask'] = mask
            uncertain_mask = self.cal_uncertain_mask(pred_logits, mask)
            self.store_values['uncertain'] = uncertain_mask
            gland_points, bg_points = self.get_points_train(gt_seg, uncertain_mask)
        # print("没进来")
        # exit(1)
# AttributeError: 'dict' object has no attribute 'shape'内容是如何确认的大小or元素类型
        # print("feats.shape:",feats.shape)
        
        # print("feats:")
        # 假设 feats 是一个列表或数组
        # print("low_level shape:",feats['low_level'].shape)
        # print("layer2 shape:",feats['layer2'].shape)
        # print("aux shape:",feats['aux'].shape)
        # print("out:",feats['out'].shape)
# 将 feats 写入一个文件
        # with open('feats_output——xxn.txt', 'w') as f:
        #     # 如果 feats 是一个列表或包含较复杂数据结构，你可以遍历并写入
        #     for key, value in feats.items():
        #         f.write(f"{key}: {value}\n")


        # print(feats)
        # exit(1)
        #gland_feats = self.point_sample(feats, gland_points, align_corners=True)
        #gland_feats = gland_feats.permute(0, 2, 1)

        #bg_feats = self.point_sample(feats, bg_points, align_corners=True)
        #bg_feats = bg_feats.permute(0, 2, 1)
        gland_feats = self.multi_level_point_sample(feats, gland_points, align_corners=True, fcs=fcs)
        bg_feats = self.multi_level_point_sample(feats, bg_points, align_corners=True, fcs=fcs)





        #dummy = torch.randn([feats.shape[0], 256, 480, 480], device=feats.device)
        #gland_feats, bg_feats = self.hard_sampling_even(feats, gt_seg, mask)
        #return 0, mask

        ####################################
        # save gt in tmp


        #print(gland_feats.shape, bg_feats.shape)


        # gland : 0
        # bg : 1

        #feats = torch.cat([
        #    gland_feats,
        #    bg_feats
        #], dim=0)
        #feats = feats.view(-1, feats.shape[-1])

        #batch_size, num_samples, dim = gland_feats.shape
        #feats = torch.nn.functional.normalize(feats, dim=1, p=2)


        #labels = torch.cat([
        #    torch.zeros([batch_size * num_samples], device=queue.device, dtype=torch.long),
        #    torch.ones([batch_size * num_samples], device=queue.device, dtype=torch.long)
        #], dim=0).view(-1)



        #print(feats.shape, labels.shape, queue.shape)
        #loss = self.constrasive(feats, labels, queue)
        #print(feats.shape, labels.shape)
        #feats = torch.nn.
        #queue_y = torch.cat([
        #    torch.zeros([1, queue.shape[1]], device=queue.device, dtype=torch.long),
        #    torch.ones([1, queue.shape[1]], device=queue.device, dtype=torch.long)
        #], dim=0).view(-1)


        #queue_feat = queue.view(-1, dim)

        #print(queue_y.shape, queue_feat.shape, feats.shape, labels.shape)
        #nce_feats = torch.cat(
        #    [
        #        feats,
        #        queue_feat
        #    ]
        #)
        #nce_labels = torch.cat(
        #    [
        #        labels,
        #        queue_y
        #    ]
        #)


        #loss = self.infonce_loss_fn(feats, labels)

        # torch.Size([8, 4, 64])
        # print(gland_feats.shape)
        gland_feats = torch.nn.functional.normalize(gland_feats.contiguous().view(-1, gland_feats.shape[-1]), dim=1, p=2)
        # print(gland_feats.shape)
        # torch.Size([32, 64])
        # exit(1)
        #print(gland_feats.shape, 'after')
        bg_feats = torch.nn.functional.normalize(bg_feats.contiguous().view(-1, bg_feats.shape[-1]), dim=1, p=2)
        #print(bg_feats.shape)

        gland_queue = queue[0]
        bg_queue = queue[1]
        # <class 'torch.Tensor'>
        # print(type(queue))
        # print(queue[0])
        # 2
        # torch.Size([1000, 64])
        # print(len(queue))
        # print(queue[0].shape)
        # exit(1)

        #zeros_short = torch.zeros([gland_feats.shape[0]], device=queue.device, dtype=torch.long)
        #ones_long = torch.ones([gland_queue.shape[0]], device=queue.device, dtype=torch.long)


        #gland_queue_labels = torch.zeros([gland_feats.shape[0]], device=queue.device, dtype=torch.long)
        #bg_queue_labels = torch.ones([bg_feats.shape[0]], device=queue.device, dtype=torch.long)
        #gland_loss = self.compute_loss(gland_feats, bg_queue, zeros_short, ones_long)
        #gland_loss = self.compute_loss(gland_feats, bg_queue)
        gland_loss = self.contrasive_single_class(anchor=gland_feats, postive=gland_queue.detach(), negative=bg_queue.detach())
        bg_loss = self.contrasive_single_class(anchor=bg_feats, postive=bg_queue.detach(), negative=gland_feats.detach())

        #return 0, uncertain_mask
        #print(bg_loss)
        #print((gland_loss + bg_loss) / 2)
        #import sys; sys.exit()


        #gland_loss = self.infonce_loss_fn(
        #    torch.cat([
        #        gland_feats,
        #        bg_queue
        #    ])
        #)
        #print(id(bg_queue), id(queue[1]))
        #import sys; sys.exit()

        #loss = self.infonce_loss_fn()

        loss = (self.temperature / self.base_temperature) * (gland_loss + bg_loss) / 2

        with torch.no_grad():
            self._dequeue_and_enqueue(gland_feats.detach(), bg_feats.detach(), queue, queue_ptr)

        #print(loss)
        #import sys; sys.exit()


        #pred_seg = pred_logits.argmax(dim=1)
        #print(pred_seg.shape)
        #print(mask.shape, gt_seg.shape)
        #print((mask == 1).shape)
        #gt_seg[mask == 1] = 0
        #print(type(pred_seg), type(mask))
        #bg_candicate = gt_seg * mask

        #bg_candidate = bg_candicate == 1
        #print(type(c))
        #import sys; sys.exit()
        #bg_candicate = mask == 1 && gt_seg == 1
        #gland_candicate = mask == 1  && gt_seg == 0




        #pred_seg[mask ]



        #pred_seg =
        #print(pred_seg.requires_grad)
        #print(pred_seg.shape)
        #pred_mask = connected_components(pred_logits, num_iterations=100)
        #a = torch.tensor([[4, 4, 0 , 0],
        #[4, 4, 0 , 2],
        #[4, 4, 0 , 2],
        #[4, 4, 0 , 0]])

        #b = torch.tensor([[3, 0, 0 , 0],
        #[3, 3, 0 , 0],
        #[0, 0, 0 , 2],
        #[4, 4, 0 , 0]])

        #print(a)
        #print(b)
        #c = a + b
        #print(c)


        #print(pred_seg.shape)
        #print(pred_seg.shape)
        #print(pred_seg)

        #return loss, mask
        return loss, uncertain_mask

#loss = GlandContrastLoss(grid_size=28, num_nagative=2)
#
#img = torch.randn(6, 256, 120, 120)
#gt_seg = torch.randn(6, 120, 120).long()
#gt_seg[gt_seg >= 0] = 1
#gt_seg[gt_seg < 0] = 0
##print(gt_seg.max(), gt_seg.min())
#queue = torch.randn(2, 30, 256)
#queue_ptr = torch.zeros(1)
##print(queue_ptr)
##print(queue.mean())
#print(loss(img, gt_seg, queue, queue_ptr))
##print(queue_ptr)
##print(queue.mean())
#import sys; sys.exit(0)
#import cv2
##img = cv2.imread('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_3_anno.bmp', -1)
##img = cv2.imread('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/cells_binary.png', 0)
##img = cv2.imread('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_3_anno.bmp', cv2.IMREAD_GRAYSCALE)
##import numpy as np
##img[img != 0] = 255
##print(img.shape)
##print(np.unique(img))
##img = torch.tensor(img).unsqueeze(0) / img.max()
##print(img.dtype)
##print(img.shape, gt_seg.shape)
#img.requires_grad=True
#cc = loss(img.cuda(), gt_seg.cuda(), queue)
#print(cc)
##print(np.unique(img))
##print(img.shape)
##c = connected_components(img, num_iterations=500)
##print(len(torch.unique(c)))
##print(torch.unique(c))
##
##c = c.squeeze()
##c = c.numpy()
##c = c / c.max() * 255
##c = c.astype('uint8')
##print(c.shape)
##cv2.imwrite('cc.jpg', c)
#import sys; sys.exit()
#
#
#
#
#
##
# from .utils import get_class_weight, weighted_loss
from .utils import weighted_loss



@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


# @LOSSES.register_module()
class DiceLoss(nn.Module):
    """DiceLoss.
    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        # self.class_weight = get_class_weight(class_weight)
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss
