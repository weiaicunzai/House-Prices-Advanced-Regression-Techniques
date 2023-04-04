import os
import sys
import math
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

#from models.backbones.resnet import resnet50d
import models.backbones.resnet as resnet
#import models.head.fcnhead as fcnhead
#from models.head.fcnhead import FCNHead, UpSample2d
#from models.dual_gcn import DualGCNHead

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous() # b x k x c
        support = torch.matmul(x_t, self.weight) # b x k x c

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous() # b x c x k

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        #self.gcn1 = GraphConvNet(dim, dim)
        #self.gcn2 = GraphConvNet(dim, dim)
        #self.gcn3 = GraphConvNet(dim, dim)
        #self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        #assert(loop == 1 or loop == 2 or loop == 3)
        #self.gcns = nn.Module()
        gcns = []
        for i in range(loop):
            gcns.append(GraphConvNet(dim, dim))
        self.gcns= nn.Sequential(*gcns)
        #self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous() # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x)) # b x c x k
        x = self.relu(x)
        return x

class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def init(self, initcache):
        if not os.path.exists(initcache):
            print(initcache + ' not exist!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.init_params(clsts, traindescs)
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H*W
        residual = (x.view(B, C, -1).permute(0, 2, 1).unsqueeze(1).contiguous() - self.anchor.unsqueeze(1)).div(sigma.unsqueeze(1)  + 1e-7)
        #residual_norm = torch.norm(residual, dim=3, p='fro')
        #residual_pow = -torch.pow(residual_norm, 2) / 2
        residual_pow = - torch.pow(torch.norm(residual, dim=3, p='fro'), 2) / 2
        soft_assign = F.softmax(residual_pow, dim=1)
        #print(soft_assign.shape)
        #import sys; sys.exit()
        return soft_assign


    def gen_soft_assign1(self, x, sigma):
        #torch.save(x, 'x.pt')
        #torch.save(sigma, 'sigma.pt')
        B, C, H, W = x.size()
        N = H*W
        #soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)
        soft_assign = []
        #tmp = []
        #print(x.view(B, C, -1).permute(0, 2, 1).contiguous())
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]  + 1e-7)
            # residual: [B, 16, num_pixels, 512]
            #print(residual.shape, 'cc')
            # x : [B, 1,  3600, 512]
            # anchor: [16, 1, 512]
            # residual [B, 16, 3600, 512]
            #soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2
            #print(11, residual.shape) [3, 3600, 512]
            #print(torch.norm(residual, dim=2).shape)
            soft_assign.append(-torch.pow(torch.norm(residual, dim=2), 2) / 2)
            #print(soft_assign[-1].shape)
            #soft_assign.append(soft_assign[:, node_id, :])
            #print(tmp[-1].shape)
            #print(soft_assign[:, node_id, :].shape)
            #print(residual_pow.max().item(), residual_pow.min().item())
            #print(residual_pow.mean().item(), residual_pow.std().item())
            #print('pow', residual_pow.max().item(), residual_pow.min().item())
            #residual_final = - residual_pow / 2
            #print('final', residual_final.max().item(), residual_final.min().item())
            #soft_assign[:, node_id, :] = residual_pow
            #print(soft_assign.dtype, residual_pow.dtype)
            # print(";;;;;;;;;;;;", soft_assign[:, node_id, :].mean().item(), soft_assign[:, node_id, :].std().item())

        #print('before max:', soft_assign.max().item())
        #print('after min:', soft_assign.min().item())
        #torch.save(soft_assign, 'soft_assign.pt')
        #print(soft_assign.shape)
        # [3, 16, 3600]
        soft_assign = torch.stack(soft_assign, dim=1)

        #print(tmp.shape)
        #print(soft_assign.shape)
        #print((soft_assign - tmp).mean())
        soft_assign = F.softmax(soft_assign, dim=1)
        #print('ccc', soft_assign.shape)
        #print('max:', soft_assign.max().item())
        #print('min:', soft_assign.min().item())
        #if self.training:
            #import sys; sys.exit()
            #if soft_assign.isnan().sum() > 0:
                #import sys; sys.exit()

        #import sys; sys.exit()
        return soft_assign


    def gen_nodes(self, x, sigma, soft_assign):
        B, C, H, W = x.size()
        #N = H*W


        # function (3)
        residual = (x.view(B, C, -1).permute(0, 2, 1).unsqueeze(1).contiguous() - self.anchor.unsqueeze(1)).div(sigma.unsqueeze(1)  + 1e-7)
        numerator = residual.mul(soft_assign.unsqueeze(-1)).sum(dim=2)
        denominator = soft_assign.sum(dim=2).unsqueeze(-1)
        nodes = numerator / (denominator + 1e-7)

        return nodes


    def forward(self, x):
        B, C, H, W = x.size()
        #if self.normalize_input:
            #x = F.normalize(x, p=2, dim=1) #across descriptor dim

        # We constrain the range of each element in σk to (0, 1)
        # by defining σk as the output of a sigmoid function.
        sigma = torch.sigmoid(self.sigma)
        #import time
        #t1 = time.time()
        soft_assign = self.gen_soft_assign(x, sigma) # B x C x N(N=HxW)
        #t2 = time.time()
        #soft_assign1 = self.gen_soft_assign1(x, sigma) # B x C x N(N=HxW)
        #t3 = time.time()
        #print(t2 - t1)
        #print(t3 - t2)
        #print((soft_assign - soft_assign1).mean())
        #print(torch.cuda.memory_summary())
        #import sys; sys.exit()
        #print(torch.isnan(x).sum().item(), soft_assign.mean().item())
        #
        #with torch.no_grad():
        nodes = self.gen_nodes(x, sigma, soft_assign)


        #eps = 1e-7
        #nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        #for node_id in range(self.node_num):
        #    residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]  + 1e-7)
        #    nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2) # intra-normalization
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1) # l2 normalize

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign


#class Attention(nn.Module):
#    def __init__(self, project_ratio=8):
#        self.project = self.neck = nn.Sequential(
#            nn.Linear(, 2048),
#            nn.ReLU(inplace=True),
#            nn.Linear(2048, 256)
#        )


class GCU(nn.Module):
    def __init__(self, node_num, dim, loop=1):
        super().__init__()
        self.project = GraphNet(node_num, dim)
        self.gcns = CascadeGCNet(dim=dim, loop=loop)
        self.r = 0.05


    def attention(self, q, k):
        b, dim, num_nodes = q.shape
        q_len, q_dim = k.shape
        q = q.permute(0, 2, 1) # [B, num_nodes, dim]
        q = nn.functional.normalize(q, p=2, dim=2)
        #k = k.T.expand(b, q_dim, q_len) # [B, dim, q_len]

        #print(q.shape, k.shape)
        sim = torch.einsum('bnc, bck->bnk', [q, k.T.expand(b, q_dim, q_len)]) / 0.07
        #print(sim.shape)
        #k = k.permute(0, )
        #print(sim.shape, k.shape, k.requires_grad)
        weighted_sim = sim.softmax(dim=2)
        _, indices = weighted_sim.topk(int(q_len * self.r), dim=2)

        # gather top k similarity matrix
        top_sim = torch.gather(input=weighted_sim, dim=2, index=indices)

        #print(indices.shape, top_sim.shape, weighted_sim.shape)
        # torch.Size([16, 8, 200]) torch.Size([16, 8, 200]) torch.Size([16, 8, 2000])
        top_k_indices = indices.unsqueeze(-1).expand(b, num_nodes, int(q_len * self.r), dim)
        #print(top_k_indices, weighted_sim)
        #print(k.shape, 'no expand')
        k_expand = k.unsqueeze(0).unsqueeze(0).expand(b, num_nodes, -1, -1)
        #print(k_expand.shape, 'expand')
        top_k = torch.gather(input=k_expand, dim=2, index=top_k_indices)
        #print(top_k, top_sim)
        #print(top_k.shape, top_sim.shape)
        weighted_top_k = top_k * top_sim.unsqueeze(-1)
        attn_top_k = weighted_top_k.sum(dim=2)
        #print(attn_top_k.shape, 'attn_top_k')
        #print(sim.shape, k.shape)
        q = attn_top_k + q
        #print(q.shape, attentio)
        q = q.permute(0, 2, 1) # [B, num_nodes, dim]
        #import sys; sys.exit()
        #weighted_sim_sum = weighted_sim.sum(dim=1)
        #q = q * weighted_sim_sum


        return q




    def forward(self, x, queue=None):
        graph, assign = self.project(x)
        graph = self.gcns(graph)

        # graph : [16, 256, 8][B, dim, num_nodes]
        # if queue is not None:
        #     attn = self.attention(graph, queue)
        #     graph = graph + attn

        graph = graph.bmm(assign)


        graph = graph.view(x.shape)

        return graph



class GraphHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #print(node_num, dim)
        #import sys; sys.exit()
        #self.project = GraphNet(node_num, dim)
        #self.gcns = CascadeGCNet(dim=dim, loop=1)
        sub_dim = int(dim / 4)
        assert sub_dim * 4 == dim
        self.gcu_2 = GCU(node_num=2, dim=sub_dim)
        self.gcu_4 = GCU(node_num=4, dim=sub_dim)
        self.gcu_8 = GCU(node_num=8, dim=sub_dim)
        self.gcu_16 = GCU(node_num=16, dim=sub_dim)
        #self.gcu_32 = GCU(node_num=32, dim=sub_dim)

        self.project = BasicConv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1
        )

        self.attention_gcu = GCU(node_num=8, dim=dim)

    #def restore(self, graph, assign):
    #    #assign = graph.permute(0, 2, 1).contiguous().bmm(region1)
    #    print(assign.shape)
    #    assign = F.softmax(assign, dim=-1) #normalize region-node
    #    m = assign.bmm(graph.permute(0, 2, 1).contiguous())
    #    m = m.permute(0, 2, 1).contiguous()
    #    return m

    def forward(self, x, queue=None):

        out = []
        inputs = x.split(64, dim=1)
        assert len(inputs) == 4
        #print(inputs[0].shape, x.shape)
        out.append(self.gcu_2(inputs[0]))
        out.append(self.gcu_4(inputs[1]))
        out.append(self.gcu_8(inputs[2]))
        out.append(self.gcu_16(inputs[3]))
        #out.append(self.gcu_32(inputs[3]))

        out = torch.cat(out, dim=1)

        if queue is not None:
            queue = queue.view(-1, queue.shape[-1])

        out = self.attention_gcu(out, queue)
        #print(out.shape)
        #print(out.shape)
        out = self.project(out)
        out = out + x

        return out
        #graph, assign = self.project(x)
        #graph = self.gcns(graph)
        #graph = graph.bmm(assign)

        #graph = graph.view(x.shape)

        #return graph
        #return spatial_x


class BasicLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Sequential(
            #nn.Linear(in_channels, in_channels),
            #nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TG(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        #self.graph_head_dim = 512
        #self.graph_head_dim = 256
        self.graph_head_dim = 256

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(self.graph_head_dim + 48, 256, 3, padding=1, bias=False),
            #nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        #q_len = 5000
        q_len = 1000
        #q_len = 2500
        q_dim = 256
        self.register_buffer("queue", torch.randn(num_classes, q_len, q_dim))
        #self.register_buffer("queue", torch.randn(num_classes, q_len, 2))
        self.queue = nn.functional.normalize(self.queue, p=2, dim=2)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.fcs =  nn.ModuleList(
            [
                BasicLinear(256, 256),
                BasicLinear(512, 256),
                BasicLinear(1024, 256),
                BasicLinear(2048, 256),
                # BasicLinear(256, 256),
            ]
        )

        #self.neck = nn.Sequential(
        #    nn.Linear(2048, 2048),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(2048, 256)
        #)
        #self.out = nn.Sequential(
        #    nn.Conv2d(self.graph_head_dim + 48, 256, 1, padding=1, bias=False),
        #    nn.BatchNorm2d(256),
        #    nn.ReLU(inplace=True),
        #)

        #self.queue = nn.Parameter(num_classes, 5000, 256)
        #self.register_buffer("queue", torch.randn(num_classes, 5000, 256))
        #self.queue = nn.functional.normalize(self.queue, p=2, dim=2)
        #self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        #self.head = FCNHead()
        #self.cls_head = UpSample2d(
        #    320,
        #    num_classes,
        #    scale_factor=2
        #)
        #self.head = DualGCNHead(
        #    2048,
        #    512,
        #    num_classes
        #)
        #self.cls_head = nn.Sequential(
        #self.gland_head = GlandHead(32, 512)

        #self.out = nn.Sequential(
        #    BasicConv2d(
        #        in_channels=512,
        #        out_channels=256,
        #        kernel_size=1
        #    ),
        #    #BasicConv2d(512, num_classes, 1)
        #)

        self.gland_head_project = nn.Sequential(
            BasicConv2d(
                in_channels=2048,
                #out_channels=512,
                out_channels=self.graph_head_dim,
                kernel_size=1
            )
        )
        self.gland_head = GraphHead(dim=self.graph_head_dim)

        self.aux_head = nn.Sequential(
            BasicConv2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=1
            ),
            BasicConv2d(512, num_classes, 1)
        )
        #self.aux_head = BasicConv2d(
        #    BasicConv2d(1024, 512),
        #    BasicConv2d(512, num_classes)
        #)
        #self.fcs = nn.


    def forward(self, x):

        B, C, H, W = x.shape


        feats = self.backbone(x)
        low_level_feat = self.project(feats['low_level'])

        gland_feats = self.gland_head_project(feats['out'])
        gland = self.gland_head(gland_feats, queue=self.queue.detach()) # layer 4


        gland = F.interpolate(
            gland,
            #size=(H, W),
            size=low_level_feat.shape[2:],
            align_corners=True,
            mode='bilinear'
        )

        #feats['gland'] = gland

        #concat_inputs = torch.cat([gland, low_level_feat], dim=1)
        gland = torch.cat([gland, low_level_feat], dim=1)


        #feats[]
        #for key, value in feats.items():
            #print(key, value.shape)




        #concat_inputs = self.out(concat_inputs)
        #gland = self.out(gland)

        #sample = gland


        #concat_inputs = F.interpolate(

       #classify before interploate
        #gland = self.classifier(output)
        #sampler = gland
        gland = self.classifier(gland)

        # output this value......
        gland = F.interpolate(
            #concat_inputs,
            gland,
            size=(H, W),
            #size=low_level_feat.shape[2:],
            align_corners=True,
            mode='bilinear'
        )




        if not self.training:
            return gland

        aux = self.aux_head(feats['aux'])
        aux = F.interpolate(
            aux,
            size=(H, W),
            align_corners=True,
            mode='bilinear'
        )


        return gland, aux, feats







        #assert out.shape[2:] == x.shape[2:]
        #assert torch.equal(out.shape[2:], x.shape[2:])


        #return output
        #return gland


def tg(num_classes):
    backbone = resnet.resnet50d()
    backbone.fc = nn.Identity()
    backbone.global_pool = nn.Identity()
    #print(backbone)
    net = TG(backbone=backbone, num_classes=num_classes)
    return net


#net = tg(2)
###print(sum(p.numel() for p in net.parameters() if p.requires_grad))
###print(sum(p.numel() for p in net.backbone.parameters() if p.requires_grad))
####print(sum([p.numel() ]))
#img = torch.randn(3, 3, 480, 480)
###
#output = net(img)



#print(output.shape)

#for x in output:
    #print(x.shape)
