import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
#from layers import GraphConvolution
from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj
from torch_geometric.nn import GCNConv
from layers import GraphConvolution

#iden_mat = torch.eye(num_nodes).to('cuda:0')

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
        self.bn = [nn.InstanceNorm1d(nhid).to('cuda') for idx in range(n_layers)]
        self.bn.append(nn.InstanceNorm1d(nclass).to('cuda'))
        self.iden_mat = iden_mat = torch.eye(num_nodes).to('cuda:0')
        self.layers = nn.ModuleList()
        list_nhid = [nhid]
        list_nhid = list_nhid*(self.nlayers-1)
        if activation == 'relu':
            self.activation= F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'step':
            self.activation = custom_step_function()
        nhid = list_nhid
        self.num_nodes = num_nodes
        #self.layers.append(GCNConv(nfeat, nhid[0]))
        self.layers.append(GraphConvolution(nfeat, nhid[0]))
        # if (len(nhid) != (self.nlayers-1)):
        #   print("Parameter nhid doesn't have correct length")
        for i in range(1, len(nhid)):
            #self.layers.append(GCNConv(nhid[i - 1], nhid[i]))
            self.layers.append(GraphConvolution(nhid[i - 1], nhid[i]))
        # self.layers.append(GCNConv(nhid[-1], nclass))
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

    


    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm
# def __init__(self, activation,n_layers, num_nodes, nfeat, nhid, nclass, dropout, device,bias=True):

class GCNII(nn.Module):
    def __init__(self,activation,n_layers,num_nodes,nfeat, nhid, nclass,dropout,device,bias,alpha, theta, shared_weights=True ):
        super(GCNII, self).__init__()

        self.name = 'GCNII'
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(nfeat, nhid))
        self.lins.append(nn.Linear(nhid, nclass))
        
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(n_layers):
            self.convs.append(
                GCN2Conv(nhid, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.activation_name = activation
        
        self.dropout = dropout
        self.device = device
        self.nhid = nhid
        self.nlayers = n_layers
        self.num_nodes = num_nodes
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        # x = data.graph['node_feat']
        x = data.x
        # n = data.graph['num_nodes']
        n = data.num_nodes
        # edge_index = data.graph['edge_index']
        edge_index = data.edge_index
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x


