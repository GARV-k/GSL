# define a single graph convolution layer
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #print(f"Initializing GraphConvolution with in_features: {in_features}, out_features: {out_features}")
#struct_weight: new trainable Weight parameter
# shape (num_nodes, num_node ) 
        #self.struct_weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.weight = Parameter(torch.FloatTensor(int(in_features), int(out_features)))
        if bias:
            self.bias = Parameter(torch.FloatTensor(int(out_features)))
        else:
            # self.register_parameter('bias', None)
            pass
        self.reset_parameters()

    # initializing weights
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # (X, W)
        support = torch.mm(input, self.weight)
        # (A, XW)
        output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'