import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import GraphConvolution
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj

def norm_adj(num_nodes, adj_matrix):
  # normalization
  # A' = A + I
  adj_matrix += torch.eye(num_nodes).to('cuda:0')
  # D' = D + I
  degrees = torch.sum(adj_matrix, dim = 1)
  degree_matrix = torch.diag(1 / torch.sqrt(degrees))
  norm_adj = torch.mm(degree_matrix, adj_matrix)
  norm_adj = torch.mm(norm_adj, degree_matrix)
  return norm_adj


class custom_step_function(nn.Module):
    def __init__(self):
        super(custom_step_function, self).__init__()
    
    def forward(self, x):
        x[x>=0] = 1.0
        x[x<0] = 0.0
        return x

class GCN_L(nn.Module):
    def __init__(self, activation,n_layers, num_nodes, nfeat, nhid, nclass, dropout, device,bias=True):
        super(GCN_L, self).__init__()
        self.device = device
        self.name = 'structured_GCN'
        self.nlayers = n_layers
        self.dropout = dropout
        self.nhid = nhid
        self.activation_name = activation
        self.layers = nn.ModuleList()
        list_nhid = [nhid]
        list_nhid = list_nhid*(self.nlayers-1)
        nhid = list_nhid
        if activation == 'relu':
            self.activation= F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'step':
            self.activation = custom_step_function()
        
        self.num_nodes = num_nodes
        self.bn = nn.BatchNorm1d(num_nodes).to('cuda')
        
        self.layers.append(GraphConvolution(nfeat, nhid[0]))
        for i in range(1, len(nhid)):
            self.layers.append(GraphConvolution(nhid[i - 1], nhid[i]))
        self.layers.append(GraphConvolution(nhid[-1], nclass))
        self.dropout = dropout
        self.struct_weight = Parameter(torch.FloatTensor(num_nodes,num_nodes))
        if bias:
            self.struct_bias = Parameter(torch.FloatTensor(num_nodes))
        else:
            pass
        self.reset_parameters()




    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.struct_weight.size(1))
        self.struct_weight.data.uniform_(-stdv, stdv)
        if self.struct_bias is not None:
            self.struct_bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        mod_adj = self.activation(torch.mm(adj, self.struct_weight) + self.struct_bias)
        #adj_final: noramlized modified adj matrix
        adj_final = norm_adj(self.num_nodes, mod_adj.clone())
        for layer in self.layers[:-1]:
            x = layer(x, adj_final)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, adj_final)
        return F.log_softmax(x, dim=1)
    



class GCN(nn.Module):
    def __init__(self, activation,n_layers, num_nodes, nfeat, nhid, nclass, dropout, device,bias=True):
        super(GCN, self).__init__()
        self.device = device
        self.name = 'normal_GCN'
        self.nlayers = n_layers
        self.nhid = nhid
        self.layers = nn.ModuleList()
        list_nhid = [nhid]
        list_nhid = list_nhid*(self.nlayers-1)
        nhid = list_nhid
        self.num_nodes = num_nodes
        self.activation_name = activation
        self.layers.append(GraphConvolution(nfeat, nhid[0]))
        # if (len(nhid) != (self.nlayers-1)):
        #   print("Parameter nhid doesn't have correct length")
        for i in range(1, len(nhid)):
            self.layers.append(GraphConvolution(nhid[i - 1], nhid[i]))
        self.layers.append(GraphConvolution(nhid[-1], nclass))
        self.dropout = dropout
        self.struct_weight = Parameter(torch.FloatTensor(num_nodes,num_nodes))
        if bias:
            self.struct_bias = Parameter(torch.FloatTensor(num_nodes))
        else:
            # self.register_parameter('bias', None)
            pass
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.struct_weight.size(1))
        self.struct_weight.data.uniform_(-stdv, stdv)
        if self.struct_bias is not None:
            self.struct_bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        mod_adj = adj
        #adj_final: noramlized modified adj matrix
        adj_final = norm_adj(self.num_nodes, mod_adj.clone())
        for layer in self.layers[:-1]:
            x = layer(x, adj_final)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, adj_final)
        return F.log_softmax(x, dim=1)
    



#Below is the implementation of GCN2 model from the PyG Documentation- Examples in Repo - gcn2_cora.py
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN2Conv
# def __init__(self, activation,n_layers, num_nodes, nfeat, nhid, nclass, dropout, device,bias=True):
class GCN2(torch.nn.Module):
    def __init__(self, activation, num_layers, num_nodes, nfeat, hidden_channels, nclass, dropout,device, bias, alpha, theta,
                 shared_weights=True,):
        super().__init__()
        self.name = 'GCN2'
        #self.edge_index = edge_index
        self.lins = torch.nn.ModuleList()
        self.nhid = hidden_channels
        self.lins.append(Linear(nfeat, hidden_channels))
        self.lins.append(Linear(hidden_channels, nclass))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout
        self.device = device
        self.activation_name = activation

    def forward(self, x, adj_t):
        edge_index, edge_weight = dense_to_sparse(adj_t)
        # Ensure edge_index is of type long
        edge_index = edge_index.long()
        adj_t = edge_index
        #print(f"shape: {adj_t.shape}")
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


