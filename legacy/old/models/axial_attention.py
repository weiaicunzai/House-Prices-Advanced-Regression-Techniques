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

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PlainCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
                BasicConv2d(in_channels, out_channels, kernel_size, **kwargs),
                BasicConv2d(out_channels, out_channels, kernel_size, **kwargs)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class CombinedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cnn_block, attn_block, branch='hybird'):
        super().__init__()
        #print(stride)
        #self.conv = BasicConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=False)
        #self.out = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.out = BasicConv2d(out_channels, out_channels, 1)
        self.cnn_branch = cnn_block
        self.attn_branch = attn_block
        self.branch = branch

    def forward(self, x):
        x = self.project(x)
        #ori_size = x.shape[-2:]
        #print(x.shape, ori_size)
        if self.branch not in ['hybird', 'trans', 'cnn']:
            raise ValueError('branch should be one of these: both, trans, cnn')

        if self.branch == 'hybird':
            #print(self.branch)
            cnn_feature = self.cnn_branch(x)
            attn_feature = self.attn_branch(x)

            if cnn_feature.shape != attn_feature.shape:
                attn_shape = attn_feature.shape[-2:]
                x = F.interpolate(x, size=attn_shape)

            feature = cnn_feature + attn_feature

        elif self.branch == 'cnn':
            #print(self.branch)
            cnn_feature = self.cnn_branch(x)
            feature = cnn_feature
        elif self.branch == 'trans':
            #print(self.branch)
            attn_feature = self.attn_branch(x)
            feature = attn_feature

        #feature = cnn_feature + attn_feature
        #feature = cnn_feature
        #print(feature.shape, self.out)
        x = self.out(feature)
        return x

#class HyberNet(nn.Module):
#    def __init__(self, num_classes):
#        super().__init__()
#        self.stem = BasicConv2d(3, 64, 3, stride=2, padding=1)
#        #kernel_sizes = [128, 64, 32, 16]
#        kernel_sizes = [118, 59, 29, 14]
#        self.layer1 = self._make_layers(64, 128, 2, kernel_size=kernel_sizes[0])
#        self.layer2 = self._make_layers(128, 256, 2, kernel_size=kernel_sizes[1])
#        self.layer3 = self._make_layers(256, 512, 2, kernel_size=kernel_sizes[2])
#        self.layer4 = self._make_layers(512, 1024, 2, kernel_size=kernel_sizes[3])
#
#        self.decoder1 = nn.Conv2d(1024, 256, 1)
#        self.decoder2 = nn.Conv2d(256, 64, 1)
#        self.cls_pred = nn.Conv2d(64, num_classes, 1)
#
#    def forward(self, x):
#        src_shape = x.shape
#        x = self.stem(x)
#        low_level = x
#        #print(x.shape, 'before layer 1')
#        x = self.layer1(x)
#        #print(x.shape, 'before layer 2')
#        x2 = self.layer2(x)
#        #print(x.shape)
#        x = self.layer3(x2)
#        #print(x.shape)
#        x = self.layer4(x)
#        #print(x.shape)
#        x = self.decoder1(x)
#        x = F.interpolate(x, size=x2.shape[-2:])
#        x = x + x2
#        x = self.decoder2(x)
#        x = F.interpolate(x, size=low_level.shape[-2:])
#        x = x + low_level
#        x = F.interpolate(x, size=src_shape[-2:])
#
#        x = self.cls_pred(x)
#
#
#        #x = self.conv1(x)
#        #x = self.conv2(x)
#
#        return x
#
#    def _make_layers(self, in_channels, out_channels, stride, kernel_size=128):
#        cnn_branch = self._make_cnn_block(out_channels, out_channels)
#        attn_branch =  self._make_attn_block(out_channels, out_channels, kernel_size=kernel_size)
#        combine = CombinedBlock(in_channels, out_channels, stride, cnn_branch, attn_branch)
#
#        return combine
#
#
#
#    def _make_cnn_block(self, in_channels, out_channels):
#        cnn_block = PlainCNN(in_channels, out_channels, 3, stride=1, padding=1)
#        return cnn_block
#
#
#    def _make_attn_block(self, in_channels, out_channels, kernel_size):
#        attn_block = AxialBlock(in_channels, out_channels, kernel_size=kernel_size, num_heads=8, branch=self.branch)
#        return attn_block


class HyberNet(nn.Module):
    def __init__(self, num_classes, branch='hybird'):
        super().__init__()
        self.stem = BasicConv2d(3, 64, 3, stride=2, padding=1)
        self.branch = branch
        #kernel_sizes = [128, 64, 32, 16]
        kernel_sizes = [118, 59, 29, 14]
        self.layer1 = self._make_layers(64, 128, 2, kernel_size=kernel_sizes[0])
        self.layer2 = self._make_layers(128, 256, 2, kernel_size=kernel_sizes[1])
        self.layer3 = self._make_layers(256, 512, 2, kernel_size=kernel_sizes[2])
        self.layer4 = self._make_layers(512, 1024, 2, kernel_size=kernel_sizes[3])

        self.decoder1 = nn.Conv2d(1024, 256, 1)
        self.decoder2 = nn.Conv2d(256, 64, 1)
        self.cls_pred = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        src_shape = x.shape
        x = self.stem(x)
        low_level = x
        #print(x.shape, 'before layer 1')
        x = self.layer1(x)
        #print(x.shape, 'before layer 2')
        x2 = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x2)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.decoder1(x)
        x = F.interpolate(x, size=x2.shape[-2:])
        x = x + x2
        x = self.decoder2(x)
        x = F.interpolate(x, size=low_level.shape[-2:])
        x = x + low_level
        x = F.interpolate(x, size=src_shape[-2:])

        x = self.cls_pred(x)


        #x = self.conv1(x)
        #x = self.conv2(x)

        return x

    def _make_layers(self, in_channels, out_channels, stride, kernel_size=128):
        cnn_branch = self._make_cnn_block(out_channels, out_channels)
        attn_branch =  self._make_attn_block(out_channels, out_channels, kernel_size=kernel_size)
        combine = CombinedBlock(in_channels, out_channels, stride, cnn_branch, attn_branch, branch=self.branch)
        return combine

    def _make_cnn_block(self, in_channels, out_channels):
        cnn_block = PlainCNN(in_channels, out_channels, 3, stride=1, padding=1)
        return cnn_block


    def _make_attn_block(self, in_channels, out_channels, kernel_size):
        attn_block = AxialBlock(in_channels, out_channels, kernel_size=kernel_size, num_heads=8)
        return attn_block



def unet_axial(num_classes, branch):
    print(branch)
    net = HyberNet(num_classes, branch=branch)
    return net

#net = unet_axial(2, branch='cnn')
#img = torch.randn(3, 3, 473, 473)
##print(net)
#res = net(img)
#print(res.shape)
#net = unet_axial(2, branch='hybird')
#res = net(img)
#print(res.shape)
#
#net = unet_axial(2, branch='trans')
#res = net(img)
#print(res.shape)

#block = AxialBlock(64, 32, kernel_size=52)

#cnn = PlainCNN(32, 64, 1)
#attn = AxialBlock(32, 32,  stride=1, kernel_size=99)

#block = CombinedBlock(64, 32, 1, None, None, None, None)
#block = BasicConv2d(64, 32, 3, stride=2, padding=1)
#block = BasicConv2d(64, 32, 3, stride=1, padding=1)
#c = torch.randn(2, 32, 99, 99)
#img = torch.randn(3, 3, 473, 473)
#net = HyberNet(2)

#res = net(img)
#print(sum([p.numel() for p in net.parameters()]))
#print(res.shape)
#block = attn
#print(block(c).shape)