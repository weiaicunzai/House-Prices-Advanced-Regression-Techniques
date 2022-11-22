import math
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.one.backbone.resnetv1 import resnet50c
from models.one.backbone.resnet import resnet50



class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attention = torch.matmul(q / self.temperature, k.transpose(2, 3))

        #print(attention.shape, 'xxxx')
        if mask is not None:
            attention = attention.masked_fill(mask==0, -1e9)

        attention = self.dropout(self.softmax(attention))
        output = torch.matmul(attention, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads

        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        #self.num_heads = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(math.sqrt(embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        residual = x

        k = self.key(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = self.query(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(0)

        q = self.attention(q, k, v).transpose(1, 2).contiguous().view(B, N, -1)

        # We apply dropout [27] to the output of each sub-layer, before it is added to the
        # sub-layer input and normalized.
        output = self.dropout(self.proj(q))
        output += residual
        output = self.layernorm(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.layernorm(x + residual)

        return x


#class TransformerEncoder(nn.Module):
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.feedforward = FeedForward(embed_dim, dropout=dropout)

    def forward(self, x):
        attn_mask = torch.full(
            (x.size(1), x.size(1)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        x = self.attention(x, mask=attn_mask)
        x = self.feedforward(x)
        return x

class TransSeg(nn.Module):
    def __init__(self, class_num, embed_dim, num_heads, dropout=0.1, num_layers=12, pretrained=False, segment=True):
        super().__init__()
        #self.backbone = resnet50c(pretrained=pretrained)
        self.backbone = resnet50c(pretrained=pretrained)
        self.layers = self._build_layers(num_layers, embed_dim, num_heads, dropout)
        #self.pos_emb = nn.Parameters()
        #self.proj = nn.Linear(2048, embed_dim)
        self.proj = nn.Linear(1024, embed_dim)
        self.conv1 = nn.Conv2d(256, 64, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 30 * 30, embed_dim))

        self.segment = segment
        #print('segment', segment)
        self.pre_training = nn.Conv2d(64, 3, 1)
        self.cls_pred = nn.Conv2d(64, class_num, 1)
        #self.cls_pred = nn.Conv2d(64, 256, 1)
        #self.cls_pred = nn.Conv2d(64, 2, 1)
        #if not segment:
            #self.pre_training = nn.Conv2d(64, 3, 1)

        #else:
        #    self.cls_pred = nn.Conv2d(64, class_num, 1)
        #self.reduce = nn.Conv2d(2)
    def set_cls_pred(self):
        self.cls_pred = nn.Conv2d(64, 2, 1)

    def _flatten(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x

    def _unflatten(self, x, h, w):
        x = x.transpose(1, 2).reshape(x.shape[0], -1, h, w)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        c1, c2, c3 = self.backbone(x)
        #print(c0.shape, c2.shape, c3.shape,c4.shape)

        x = self._flatten(c3)
        x = self.proj(x) + self.pos_embed
        #print(x.shape)

        for layer in self.layers:
            x = layer(x)
        x = self._unflatten(x, c3.shape[2], c3.shape[3])

        #print('x')
        x = F.interpolate(x, size=c1.shape[2:])
        #print('x')
        x = self.conv1(x + c1)
        #print('x')
        x = F.interpolate(x, size=(H, W))
        #print('x')
        if self.segment:
            x = self.cls_pred(x)
        else:
            x = self.pre_training(x)
            x = x.reshape(x.size(0), 3, x.size(2), x.size(3))
            #x = x.argmax(dim=2)
            #print(x.shape, 1111)

        #print('x')
        return x


    def _build_layers(self, num_layers, embed_dim, num_heads, dropout):
        layers = []
        for _ in range(num_layers):
            layers.append(Block(
                embed_dim,
                num_heads,
                dropout=dropout
            ))
        return nn.Sequential(*layers)


def transseg(num_classes, segment):
    net = TransSeg(num_classes, 256, 8, segment=segment)
    return net


#net = transseg(2, True)
#torch.load('/data/by/House-Prices-Advanced-Regression-Techniques/test.pth')
#net.load_state_dict(torch.load('/data/by/House-Prices-Advanced-Regression-Techniques/test.pth'))

#a = torch.randn(3, 3, 473, 473)
##
#print(a.shape)
##head = TransSeg(2, 256, 8)
#net = transseg(2)
##print(head)
#print(sum([p.numel() for p in net.parameters()]))
##head(a)
#pred = net(a)
#pred = pred.argmax(dim=1)
#print(pred.shape)
#