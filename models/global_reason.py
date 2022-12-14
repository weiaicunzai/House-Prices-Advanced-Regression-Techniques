#import os
import math

import torch
from torch import nn
import torch.nn.functional as F

class GraphProjection(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        #super(GraphNet, self).__init__()
        super().__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    #def init(self, initcache):
    #    if not os.path.exists(initcache):
    #        print(initcache + ' not exist!!!\n')
    #    else:
    #        with h5py.File(initcache, mode='r') as h5:
    #            clsts = h5.get("centroids")[...]
    #            traindescs = h5.get("descriptors")[...]
    #            self.init_params(clsts, traindescs)
    #            del clsts, traindescs

    #def init_params(self, clsts, traindescs=None):
    #    self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H*W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1)

        return soft_assign

    def forward(self, x):
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1) #across descriptor dim

        sigma = torch.sigmoid(self.sigma)
        soft_assign = self.gen_soft_assign(x, sigma) # B x C x N(N=HxW)
        # soft
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2) # intra-normalization
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1) # l2 normalize

        # soft_assign
        # soft
        # print(soft_assign)

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign

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

    #def forward(self, x, adj):
    def forward(self, x):
        # print(x.shape)
        x_t = x.permute(0, 2, 1).contiguous() # b x k x c
        support = torch.matmul(x_t, self.weight) # b x k x c


        #x_t = x.permute(0, 2, 1).contiguous() # b x k x c

        adj = torch.matmul(x_t, x)
        adj = torch.softmax(adj, dim=2)


        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous() # b x c x k

        if self.bias is not None:
            output = output + self.bias
        #else:
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNs(nn.Module):
    def __init__(self, gcn, dim, loop):
        super().__init__()
        self.gcns = nn.Sequential(*[
            gcn(dim, dim) for _ in range(loop)
        ])
        assert loop >= 1
        #assert(loop == 1 or loop == 2 or loop == 3)
        self.relu = nn.ReLU()

    #def construct_gcns(self):
        #for _ in range(self.loops):
            #self.



    def forward(self, x):
        """x: shape  b x num_nodes x node_dim"""
        #for gcn in self.gcns:
            #print(1111)
            #x_t = x.permute(0, 2, 1).contiguous() # b x k x c
            #x = gcn(x, adj=torch.matmul(x_t, x)) # b x c x k
        x = self.gcns(x)
        x = self.relu(x)
        return x

#class GraphConv(nn.Module):
#    def __init__(self, num_gcns, gcn, num_nodes, dim, loops):
#        super().__init__()
#        self.projection = GraphProjection(num_nodes, dim)
#        self.gcns = GCNs(gcn, num_nodes, loops)
#
#    def forward(self, x):
#        x, assign =



#num_nodes = 32
#dim = 128
#
##img = torch.randn(16, 128, 32, 32)
#img = torch.randn(16, dim, 32, 32)
#
#net = GraphProjection(num_nodes, dim)
#gcns = GCNs(GraphConvNet, num_nodes, 3)
#
#x, assign = net(img)
#
#x = x.permute(0, 2, 1)
#
#x = gcns(x)
#
#x = x.permute(0, 2, 1)
#x = x.bmm(assign)
#x = x.view(x.shape[0], x.shape[1], 32, 32)
#print(x.shape)
#out,  soft_assign = net(img)
#print(soft_assign.sum(dim=1))
#print(out.shape)
#print(soft_assign.shape)

#ss = torch.softmax(soft_assign,dim=2)
#print(ss.sum(dim=2).shape)
#print(ss.sum(dim=2)[0])