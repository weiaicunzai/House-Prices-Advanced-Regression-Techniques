import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from losses import GlandContrastLoss




def test_compute_sample_weight():

    gt_seg = [
        [[0, 0, 1],
        [0, 1, 1],
        [0, 255, 255]],

        [[1, 1, 255],
        [0, 0, 0],
        [0, 0, 0]],

        [[1, 1, 255],
        [0, 1, 0],
        [1, 255, 0]],
        ]

    gt_seg = torch.tensor(gt_seg).long()
    uncertain_mask = [
        [
        [0.33, 0.1, 0],
        [0,    0,   0.77],
        [0,    0.11, 0]],

        [[1, 0.3, 0],
        [0, 0, 0],
        [0, 0.191, 0]],

        [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    ]
    uncertain_mask = torch.tensor(uncertain_mask)

    #print(gt_seg.dtype)
    #print(uncertain_mask.dtype)
    loss = GlandContrastLoss(2, ignore_idx=255)
    gland_weight = loss.compute_sample_weight(gt_seg, uncertain_mask, class_id=1)
    bg_weight = loss.compute_sample_weight(gt_seg, uncertain_mask, class_id=0)

    gland_res = [
        [[0, 0, 0],
        [0, 0, 0.77],
        [0, 0, 0]],

        [[1, 0.3, 0],
        [0, 0, 0],
        [0, 0, 0]],

        [[1, 1, 0],
        [0, 1, 0],
        [1, 0, 0]],
    ]

    bg_res = [
        [[0.33, 0.1, 0],
        [0, 0, 0],
        [0, 0, 0]],

        [[0, 0, 0],
        [0, 0, 0],
        [0, 0.191, 0]],

        [[0, 0, 0],
        [1, 0, 1],
        [0, 0, 1]],
    ]

    gland_res = torch.tensor(gland_res)
    bg_res = torch.tensor(bg_res)

    assert torch.equal(gland_weight, gland_res)
    assert torch.equal(bg_weight, bg_res)


def test_multinomial():

    gland_weight = torch.tensor([
        [[0, 0, 0],
        [0, 0, 0.77],
        [0, 0, 0]],

        [[1, 0.3, 0],
        [0, 0, 0],
        [0, 0, 0]],

        [[1, 1, 0],
        [0, 1, 0],
        [1, 0, 0]],
    ])

    bg_weight = torch.tensor([
        [[0.33, 0.1, 0],
        [0, 0, 0],
        [0, 0, 0]],

        [[0, 0, 0],
        [0, 0, 0],
        [0, 0.191, 0]],

        [[0, 0, 0],
        [1, 0, 1],
        [0, 0, 1]],
    ])
    #print(gland_weight.shape)

    gland_indices = torch.multinomial(gland_weight.view(gland_weight.shape[0], -1), 2, replacement=False)
    bg_indices = torch.multinomial(bg_weight.view(bg_weight.shape[0], -1), 2, replacement=False)
    return gland_indices, bg_indices
    #shift = 9 * torch.arange(3)
    #print(gland_indices)
    #print(bg_indices)
    #gland_indices += shift[:, None]
    #bg_indices += shift[:, None]
    #print(gland_weight.view(-1)[gland_indices.view(-1)])
    #print(bg_weight.view(-1)[bg_indices.view(-1)])

def test_indices_to_points():

    gland_weight = torch.tensor([
        [[0, 0, 0],
        [0, 0, 0.77],
        [0, 0, 0]],

        [[1, 0.3, 0],
        [0, 0, 0],
        [0, 0, 0]],

        [[1, 1, 0],
        [0, 1, 0],
        [1, 0, 0]],
    ])

    bg_weight = torch.tensor([
        [[0.33, 0.1, 0],
        [0, 0, 0],
        [0, 0, 0]],

        [[0, 0, 0],
        [0, 0, 0],
        [0, 0.191, 0]],

        [[0, 0, 0],
        [1, 0, 1],
        [0, 0, 1]],
    ])

    gland_indices, bg_indices = test_multinomial()
    #print(gland_weight)
    print(bg_weight)
    #print(gland_indices)
    loss = GlandContrastLoss(2, ignore_idx=255)
    gland_points = loss.indices_to_points(gland_indices, 3, 3)
    bg_points = loss.indices_to_points(bg_indices, 3, 3)
    shift = 9 * torch.arange(3)
    #gland_indices += shift[:, None]
    #print(gland_weight.view(-1)[gland_indices.view(-1)])

    gland_weight = gland_weight.unsqueeze(1) # [b, 1, h, w]
    bg_weight = bg_weight.unsqueeze(1) # [b, 1, h, w]

    #print(gland_indices)
    print(bg_indices)
    sampled_weight = loss.point_sample(gland_weight, gland_points, align_corners=True)
    sampled_weight = loss.point_sample(bg_weight, bg_points, align_corners=True)
    #sampled_weight = loss.point_sample(gland_weight[2].unsqueeze(0), test.unsqueeze(0), align_corners=True)
    print(sampled_weight)





#test_compute_sample_weight()
#test_multinomial()
test_indices_to_points()