from itertools import repeat
import collections.abc
from functools import partial
from typing import List, Tuple
from copy import deepcopy
import os

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import HASH_REGEX, download_url_to_file, urlparse
try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir





def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)

def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]



def drop_block_2d(
        x, drop_prob: float = 0.1, block_size: int = 7, gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x

def drop_block_fast_2d(
        x: torch.Tensor, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    block_mask = torch.empty_like(x).bernoulli_(gamma)
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.empty_like(x).normal_()
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-6)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x

#def create_attn(attn_type, channels, **kwargs):
#    module_cls = get_attn(attn_type)
#    if module_cls is not None:
#        # NOTE: it's expected the first (positional) argument of all attention layers is the # input channels
#        return module_cls(channels, **kwargs)
#    return None

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        #self.se = create_attn(attn_layer, outplanes)
        self.se = None

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(
            self,
            drop_prob: float = 0.1,
            block_size: int = 7,
            gamma_scale: float = 1.0,
            with_noise: bool = False,
            inplace: bool = False,
            batchwise: bool = False,
            fast: bool = True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)

def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        #stride = 1 if stage_idx == 0 else 2
        stride = 2 if stage_idx == 1 else 1

        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])

def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)

class FastAdaptiveAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3), keepdim=not self.flatten)

def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)

class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)

class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveCatAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)

def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1

class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
        elif pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(flatten)
            self.flatten = nn.Identity()
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'avgmax':
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == 'catavgmax':
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'
def _create_pool(num_features, num_classes, pool_type='avg', use_conv=False):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=flatten_in_pool)
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features

def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc

def create_classifier(num_features, num_classes, pool_type='avg', use_conv=False):
    global_pool, num_pooled_features = _create_pool(num_features, num_classes, pool_type, use_conv=use_conv)
    fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
    return global_pool, fc

class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net
    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering
    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.
    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample
    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled
    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled
    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    Parameters
    ----------
    block : Block, class for the residual block. Options are BasicBlockGl, BottleneckGl.
    layers : list of int, number of layers in each block
    num_classes : int, default 1000, number of classification classes.
    in_chans : int, default 3, number of input (color) channels.
    output_stride : int, default 32, output stride of the network, 32, 16, or 8.
    global_pool : str, Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    cardinality : int, default 1, number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64, factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64, number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
    block_reduce_first : int, default 1
        Reduction factor for first convolution output width of residual blocks, 1 for all archs except senets, where 2
    down_kernel_size : int, default 1, kernel size of residual block downsample path, 1x1 for most, 3x3 for senets
    avg_down : bool, default False, use average pooling for projection skip connection between stages/downsample.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0. Dropout probability before classifier, for training
    """

    def __init__(
            self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1,
            down_kernel_size=1, avg_down=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None):

        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        #out.append(x)

        x = self.maxpool(x)

        #if self.grad_checkpointing and not torch.jit.is_scripting():
        #    x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        #else:
        x1 = self.layer1(x)
        #out.append(x1)

        x2 = self.layer2(x1)
        #out.append(x2)

        x3 = self.layer3(x2)
        #out.append(x3)

        x4 = self.layer4(x3)
        out.append(x4)

        #return [x, x1, x3, x4]
        return out


        #return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        #output = self.forward_head(x[-1])
        #return output, x
        return x

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    # ResNet and Wide ResNet
    'resnet10t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet10t_176_c3-f3215ab1.pth',
        input_size=(3, 176, 176), pool_size=(6, 6),
        test_crop_pct=0.95, test_input_size=(3, 224, 224),
        first_conv='conv1.0'),
    'resnet14t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet14t_176_c3-c4ed2c37.pth',
        input_size=(3, 176, 176), pool_size=(6, 6),
        test_crop_pct=0.95, test_input_size=(3, 224, 224),
        first_conv='conv1.0'),
    'resnet18': _cfg(url='https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    'resnet18d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'resnet34d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet26': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth',
        interpolation='bicubic'),
    'resnet26d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet26t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet26t_256_ra2-6f6fa748.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.94),
    'resnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet50d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet50t': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnet101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet101d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet152': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnet152d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet200': _cfg(url='', interpolation='bicubic'),
    'resnet200d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)),
    'tv_resnet34': _cfg(url='https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
    'tv_resnet50': _cfg(url='https://download.pytorch.org/models/resnet50-19c8e357.pth'),
    'tv_resnet101': _cfg(url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
    'tv_resnet152': _cfg(url='https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
    'wide_resnet50_2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth',
        interpolation='bicubic'),
    'wide_resnet101_2': _cfg(url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'),

    # ResNets w/ alternative norm layers
    'resnet50_gn': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth',
        crop_pct=0.94, interpolation='bicubic'),

    # ResNeXt
    'resnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnext50d_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'resnext101_32x4d': _cfg(url=''),
    'resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth'),
    'resnext101_64x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnext101_64x4d_c-0d0e0cc0.pth',
        interpolation='bicubic', crop_pct=1.0,  test_input_size=(3, 288, 288)),
    'tv_resnext50_32x4d': _cfg(url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth'),

    #  ResNeXt models - Weakly Supervised Pretraining on Instagram Hashtags
    #  from https://github.com/facebookresearch/WSL-Images
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'ig_resnext101_32x8d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth'),
    'ig_resnext101_32x16d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth'),
    'ig_resnext101_32x32d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth'),
    'ig_resnext101_32x48d': _cfg(url='https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth'),

    #  Semi-Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'ssl_resnet18':  _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth'),
    'ssl_resnet50':  _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth'),
    'ssl_resnext50_32x4d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth'),
    'ssl_resnext101_32x4d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth'),
    'ssl_resnext101_32x8d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth'),
    'ssl_resnext101_32x16d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth'),

    #  Semi-Weakly Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'swsl_resnet18': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth'),
    'swsl_resnet50': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth'),
    'swsl_resnext50_32x4d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth'),
    'swsl_resnext101_32x4d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth'),
    'swsl_resnext101_32x8d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth'),
    'swsl_resnext101_32x16d': _cfg(
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth'),

    #  Efficient Channel Attention ResNets
    'ecaresnet26t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet26t_ra2-46609757.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320)),
    'ecaresnetlight': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnetlight-75a9c627.pth',
        interpolation='bicubic'),
    'ecaresnet50d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet50d-93c81e3b.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'ecaresnet50d_pruned': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet50d_p-e4fa23c2.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'ecaresnet50t': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet50t_ra2-f7ac63c4.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320)),
    'ecaresnet101d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet101d-153dad65.pth',
        interpolation='bicubic', first_conv='conv1.0'),
    'ecaresnet101d_pruned': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet101d_p-9e74cb91.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'ecaresnet200d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), crop_pct=0.94, pool_size=(8, 8)),
    'ecaresnet269d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet269d_320_ra2-7baa55cb.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 320, 320), pool_size=(10, 10),
        crop_pct=1.0, test_input_size=(3, 352, 352)),

    #  Efficient Channel Attention ResNeXts
    'ecaresnext26t_32x4d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'ecaresnext50t_32x4d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),

    #  Squeeze-Excitation ResNets, to eventually replace the models in senet.py
    'seresnet18': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet34': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth',
        interpolation='bicubic'),
    'seresnet50t': _cfg(
        url='',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'seresnet101': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet152': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnet152d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet152d_ra2-04464dd2.pth',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=1.0, test_input_size=(3, 320, 320)
    ),
    'seresnet200d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), crop_pct=0.94, pool_size=(8, 8)),
    'seresnet269d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0', input_size=(3, 256, 256), crop_pct=0.94, pool_size=(8, 8)),

    #  Squeeze-Excitation ResNeXts, to eventually replace the models in senet.py
    'seresnext26d_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26d_32x4d-80fa48a3.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'seresnext26t_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26tn_32x4d-569cb627.pth',
        interpolation='bicubic',
        first_conv='conv1.0'),
    'seresnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext50_32x4d_racm-a304a460.pth',
        interpolation='bicubic'),
    'seresnext101_32x4d': _cfg(
        url='',
        interpolation='bicubic'),
    'seresnext101_32x8d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101_32x8d_ah-e6bc4c0a.pth',
        interpolation='bicubic', test_input_size=(3, 288, 288), crop_pct=1.0),
    'seresnext101d_32x8d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101d_32x8d_ah-191d7b94.pth',
        interpolation='bicubic', first_conv='conv1.0', test_input_size=(3, 288, 288), crop_pct=1.0),

    'senet154': _cfg(
        url='',
        interpolation='bicubic',
        first_conv='conv1.0'),

    # ResNets with anti-aliasing / blur pool
    'resnetblur18': _cfg(
        interpolation='bicubic'),
    'resnetblur50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnetblur50-84f4748f.pth',
        interpolation='bicubic'),
    'resnetblur50d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetblur101d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetaa50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetaa50_a1h-4cf422b3.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0, interpolation='bicubic'),
    'resnetaa50d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetaa101d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'seresnetaa50d': _cfg(
        url='',
        interpolation='bicubic', first_conv='conv1.0'),
    'seresnextaa101d_32x8d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnextaa101d_32x8d_ah-83c8ae12.pth',
        interpolation='bicubic', first_conv='conv1.0', test_input_size=(3, 288, 288), crop_pct=1.0),

    # ResNet-RS models
    'resnetrs50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs50_ema-6b53758b.pth',
        input_size=(3, 160, 160), pool_size=(5, 5), crop_pct=0.91, test_input_size=(3, 224, 224),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs101_i192_ema-1509bbf6.pth',
        input_size=(3, 192, 192), pool_size=(6, 6), crop_pct=0.94, test_input_size=(3, 288, 288),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs152': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs152_i256_ema-a9aff7f9.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs200': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnetrs200_c-6b698b88.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs270': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs270_ema-b40e674c.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 352, 352),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs350': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs350_i256_ema-5a1aa8f1.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0, test_input_size=(3, 384, 384),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs420': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs420_ema-972dee69.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, test_input_size=(3, 416, 416),
        interpolation='bicubic', first_conv='conv1.0'),
}

#_model_pretrained_cfgs = dict()
#
#def get_pretrained_cfg(model_name):
#    if model_name in _model_pretrained_cfgs:
#        return deepcopy(_model_pretrained_cfgs[model_name])
#    return {}

#def resolve_pretrained_cfg(variant: str, pretrained_cfg=None):
#    if pretrained_cfg and isinstance(pretrained_cfg, dict):
#        # highest priority, pretrained_cfg available and passed as arg
#        return deepcopy(pretrained_cfg)
#    # fallback to looking up pretrained cfg in model registry by variant identifier
#    pretrained_cfg = get_pretrained_cfg(variant)
#    if not pretrained_cfg:
#        _logger.warning(
#            f"No pretrained configuration specified for {variant} model. Using a default."
#            f" Please add a config to the model pretrained_cfg registry or pass explicitly.")
#        pretrained_cfg = dict(
#            url='',
#            num_classes=1000,
#            input_size=(3, 224, 224),
#            pool_size=None,
#            crop_pct=.9,
#            interpolation='bicubic',
#            first_conv='',
#            classifier='',
#        )
#    return pretrained_cfg
def get_cache_dir(child_dir=''):
    """
    Returns the location of the directory where models are cached (and creates it if necessary).
    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        #_logger.warning('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')
        print('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    hub_dir = get_dir()
    child_dir = () if not child_dir else (child_dir,)
    model_dir = os.path.join(hub_dir, 'checkpoints', *child_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def download_cached_file(url, check_hash=True, progress=False):
    if isinstance(url, (list, tuple)):
        url, filename = url
    else:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(get_cache_dir(), filename)
    if not os.path.exists(cached_file):
        #_logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        print('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file

def build_model_with_cfg(
        model_cls,
        #variant,
        pretrained,
        pretrained_cfg=None,
        model_cfg=None,
        num_classes=1000,
        **kwargs
        ):
        #feature_cfg=None,
        #pretrained_strict=True,
        #pretrained_filter_fn=None,
        #pretrained_custom_load=False,
        #kwargs_filter=None,
        #**kwargs):
    # Build model with specified default_cfg and optional model_cfg
    #This helper fn aids in the construction of a model including:
    #  * handling default_cfg and associated pretrained weight loading
    #  * passing through optional model_cfg for models with config based arch spec
    #  * features_only model adaptation
    #  * pruning config / model adaptation
    #Args:
    #    model_cls (nn.Module): model class
    #    variant (str): model variant name
    #    pretrained (bool): load pretrained weights
    #    pretrained_cfg (dict): model's pretrained weight/task config
    #    model_cfg (Optional[Dict]): model's architecture config
    #    feature_cfg (Optional[Dict]: feature extraction adapter config
    #    pretrained_strict (bool): load pretrained weights strictly
    #    pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
    #    pretrained_custom_load (bool): use custom load fn, to load numpy or other non PyTorch weights
    #    kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
    #    **kwargs: model args passed through to model __init__


    #pruned = kwargs.pop('pruned', False)
    #features = False
    #feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    #pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=pretrained_cfg)
    #update_pretrained_cfg_and_kwargs(pretrained_cfg, kwargs, kwargs_filter)
    #pretrained_cfg.setdefault('architecture', variant)

    # Setup for feature extraction wrapper done at end of this fn
    #if kwargs.pop('features_only', False):
    #    features = True
    #    feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
    #    if 'out_indices' in kwargs:
    #        feature_cfg['out_indices'] = kwargs.pop('out_indices')
    #pretrained_cfg = kwargs.get('pretrained_cfg', None)
    #kwargs.pop('pretrained_cfg')

    # Build the model
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    #pretrained_url = pretrained_cfg.get('url', None)
    #pretrained_file = pretrained_cfg.get('file', None)
    #hf_hub_id = pretrained_cfg.get('hf_hub_id', None)
    #print()

    #if pruned:
        #model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    #num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    url = pretrained_cfg.get('url', None)
    if pretrained:
        cached_file = download_cached_file(url)
        #model.load_(cached_file)
        model.load_state_dict(torch.load(cached_file))
        #print(model.fc.weight.mean())

    #print(url)
    #if pretrained:

    #if pretrained:
    #    if pretrained_custom_load:
    #        # FIXME improve custom load trigger
    #        load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
    #    else:
    #        load_pretrained(
    #            model,
    #            pretrained_cfg=pretrained_cfg,
    #            num_classes=num_classes_pretrained,
    #            in_chans=kwargs.get('in_chans', 3),
    #            filter_fn=pretrained_filter_fn,
    #            strict=pretrained_strict)

    # Wrap the model in a feature extraction module if enabled
    #if features:
    #    feature_cls = FeatureListNet
    #    if 'feature_cls' in feature_cfg:
    #        feature_cls = feature_cfg.pop('feature_cls')
    #        if isinstance(feature_cls, str):
    #            feature_cls = feature_cls.lower()
    #            if 'hook' in feature_cls:
    #                feature_cls = FeatureHookNet
    #            elif feature_cls == 'fx':
    #                feature_cls = FeatureGraphNet
    #            else:
    #                assert False, f'Unknown feature class {feature_cls}'
    #    model = feature_cls(model, **feature_cfg)
    #    model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back default_cfg
    #    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    if num_classes != pretrained_cfg.get('num_classes', None):
        model.reset_classifier(num_classes)

    #if num_classes !=
    return model


def _create_resnet(pretrained=False, pretrained_cfg=None, num_classes=1000,  **kwargs):
    return build_model_with_cfg(model_cls=ResNet, pretrained=pretrained, pretrained_cfg=pretrained_cfg, num_classes=num_classes, **kwargs)

#def resnet50d(pretrained=)

#def resnet50d(pretrained=False, **kwargs):
def resnet50d(pretrained=True, num_classes=1000, **kwargs):
    """Constructs a ResNet-50-D model.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)

    return _create_resnet(pretrained=pretrained, pretrained_cfg=default_cfgs['resnet50d'], num_classes=num_classes, **model_args)

def resnet50(pretrained=True, num_classes=1000, **kwargs):
    """Constructs a ResNet-50 model.
    """
    #model_args = dict(
    #    block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)

    return _create_resnet(pretrained=pretrained, pretrained_cfg=default_cfgs['resnet50'], num_classes=num_classes, **model_args)


def resnet101(pretrained=True, num_classes=1000, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    #return _create_resnet('resnet101', pretrained, **model_args)
    return _create_resnet(pretrained=pretrained, pretrained_cfg=default_cfgs['resnet101'], num_classes=num_classes, **model_args)


def resnet101d(pretrained=True, num_classes=1000, **kwargs):
#def resnet101d(pretrained=False, **kwargs):
    """Constructs a ResNet-101-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True, **kwargs)
    #return _create_resnet('resnet101d', pretrained, **model_args)
    return _create_resnet(pretrained=pretrained, pretrained_cfg=default_cfgs['resnet101d'], num_classes=num_classes, **model_args)


# def resnet50(pretrained=True, pretrained_cfg=default_cfgs['resnet50'], **kwargs):
#     """Constructs a ResNet-50 model.
#     """
#     model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
#     return _create_resnet('resnet50', pretrained, **model_args)


#resnet = resnet50d(pretrained=True, pretrained_cfg=default_cfgs['resnet50d'])


#print()
#for i in default_cfgs.keys():
#    #print(default_cfgs[])
#    #print(i)
#    if i == 'resnet50d':
#        print(default_cfgs[i])

#print(default_cfgs['resnet50d'])
#resnet = resnet101d(pretrained=True, num_classes=100)
#print(resnet)
#img = torch.randn(3, 3, 224, 224)
#print(resnet.fc.weight.mean())
#logits, feats = resnet(img)
#print(logits.shape)
