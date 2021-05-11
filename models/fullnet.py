"""
This script defines the structure of FullNet

Author: Hui Qu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvLayer, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, bias=False, groups=groups))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('bn', nn.BatchNorm2d(out_channels))


# --- different types of layers --- #
class BasicLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BasicLayer, self).__init__()
        self.conv = ConvLayer(in_channels, growth_rate, kernel_size=3, stride=1, padding=dilation,
                              dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BottleneckLayer, self).__init__()

        inter_planes = growth_rate * 4
        self.conv1 = ConvLayer(in_channels, inter_planes, kernel_size=1, padding=0)
        self.conv2 = ConvLayer(inter_planes, growth_rate, kernel_size=3, padding=dilation, dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


# --- dense block structure --- #
class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, layer_type, dilations):
        super(DenseBlock, self).__init__()
        for i in range(len(dilations)):
            layer = layer_type(in_channels+i*growth_rate, growth_rate, drop_rate, dilations[i])
            self.add_module('denselayer{:d}'.format(i+1), layer)


def choose_hybrid_dilations(n_layers, dilation_schedule, is_hybrid):
    import numpy as np
    # key: (dilation, n_layers)
    HD_dict = {(1, 4): [1, 1, 1, 1],
               (2, 4): [1, 2, 3, 2],
               (4, 4): [1, 2, 5, 9],
               (8, 4): [3, 7, 10, 13],
               (16, 4): [13, 15, 17, 19],
               (1, 6): [1, 1, 1, 1, 1, 1],
               (2, 6): [1, 2, 3, 1, 2, 3],
               (4, 6): [1, 2, 3, 5, 6, 7],
               (8, 6): [2, 5, 7, 9, 11, 14],
               (16, 6): [10, 13, 16, 17, 19, 21]}

    dilation_list = np.zeros((len(dilation_schedule), n_layers), dtype=np.int32)

    for i in range(len(dilation_schedule)):
        dilation = dilation_schedule[i]
        if is_hybrid:
            dilation_list[i] = HD_dict[(dilation, n_layers)]
        else:
            dilation_list[i] = [dilation for k in range(n_layers)]

    return dilation_list


class FullNet(nn.Module):
    def __init__(self, color_channels, output_channels=2, n_layers=6, growth_rate=24, compress_ratio=0.5,
                 drop_rate=0.1, dilations=(1,2,4,8,16,4,1), is_hybrid=True, layer_type='basic'):
        super(FullNet, self).__init__()
        if layer_type == 'basic':
            layer_type = BasicLayer
        else:
            layer_type = BottleneckLayer

        # 1st conv before any dense block
        in_channels = 24
        self.conv1 = ConvLayer(color_channels, in_channels, kernel_size=3, padding=1)

        self.blocks = nn.Sequential()
        n_blocks = len(dilations)

        dilation_list = choose_hybrid_dilations(n_layers, dilations, is_hybrid)

        for i in range(n_blocks):  # no trans in last block
            block = DenseBlock(in_channels, growth_rate, drop_rate, layer_type, dilation_list[i])
            self.blocks.add_module('block%d' % (i+1), block)
            num_trans_in = int(in_channels + n_layers * growth_rate)
            num_trans_out = int(math.floor(num_trans_in * compress_ratio))
            trans = ConvLayer(num_trans_in, num_trans_out, kernel_size=1, padding=0)
            self.blocks.add_module('trans%d' % (i+1), trans)
            in_channels = num_trans_out

        # final conv
        self.conv2 = nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        print(x.shape)
        out = self.conv1(x)
        print(x.shape)
        out = self.blocks(out)
        print(x.shape)
        out = self.conv2(out)
        print(x.shape)
        return out


class FCN_pooling(nn.Module):
    """same structure with FullNet, except that there are pooling operations after block 1, 2, 3, 4
    and upsampling after block 5, 6
    """
    def __init__(self, color_channels, output_channels=2, n_layers=6, growth_rate=24, compress_ratio=0.5,
                 drop_rate=0.1, dilations=(1,2,4,8,16,4,1), hybrid=1, layer_type='basic'):
        super(FCN_pooling, self).__init__()
        if layer_type == 'basic':
            layer_type = BasicLayer
        else:
            layer_type = BottleneckLayer

        # 1st conv before any dense block
        in_channels = 24
        self.conv1 = ConvLayer(color_channels, in_channels, kernel_size=3, padding=1)

        self.blocks = nn.Sequential()
        n_blocks = len(dilations)

        dilation_list = choose_hybrid_dilations(n_layers, dilations, hybrid)

        for i in range(7):
            block = DenseBlock(in_channels, growth_rate, drop_rate, layer_type, dilation_list[i])
            self.blocks.add_module('block{:d}'.format(i+1), block)
            num_trans_in = int(in_channels + n_layers * growth_rate)
            num_trans_out = int(math.floor(num_trans_in * compress_ratio))
            trans = ConvLayer(num_trans_in, num_trans_out, kernel_size=1, padding=0)
            self.blocks.add_module('trans{:d}'.format(i+1), trans)
            if i in range(0, 4):
                self.blocks.add_module('pool{:d}'.format(i+1), nn.MaxPool2d(kernel_size=2, stride=2))
            elif i in range(4, 6):
                self.blocks.add_module('upsample{:d}'.format(i + 1), nn.UpsamplingBilinear2d(scale_factor=4))
            in_channels = num_trans_out

        # final conv
        self.conv2 = nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.conv2(out)
        return out



import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        #print('query:', query_index)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        #print('key_index:', key_index)
        relative_index = key_index - query_index + kernel_size - 1
        #print('relative', relative_index.shape)
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)
        #print(x.shape, '......................................')

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        #print('all', all_embeddings.shape, self.relative.shape, self.flatten_index)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        #print(q.shape, q_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):

    #def __init__(self, inplanes, planes, stride=1, groups=1,
    def __init__(self, in_channels, out_channels, stride=1, num_heads=1,
                 dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        inter_channels = int(in_channels / 2)
        self.conv_down = conv1x1(in_channels, inter_channels)
        self.bn1 = norm_layer(inter_channels)
        self.hight_block = AxialAttention(inter_channels, inter_channels, groups=num_heads, kernel_size=kernel_size)
        self.width_block = AxialAttention(inter_channels, inter_channels, groups=num_heads, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(inter_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            #self.shortcut = nn.Conv2d(planes, planes * self.expansion, kernel_size=stride, stride=stride)
            self.shortcut = nn.Conv2d(inter_channels, out_channels, kernel_size=stride, stride=stride)

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        #if self.downsample is not None:
            #identity = self.downsample(x)
        identity = self.shortcut(x)

        #print(out.shape, identity.shape)
        out += identity
        out = self.relu(out)

        return out


def fullnet(branch='both'):
    net = FullNet(color_channels=3, output_channels=3, n_layers=6, growth_rate=24, compress_ratio=0.5, drop_rate=0.1, dilations=(1,2,4,8,16,4,1), is_hybrid=True, layer_type='basic')
    return net

#import torch
#img = torch.randn(3, 3, 300, 300).cuda()
#net = fullnet().cuda()
#res = net(img)
#print(sum([p.numel() for p in net.parameters()]))
#print(res.shape)