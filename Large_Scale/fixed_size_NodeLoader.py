from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch_geometric.nn
import torch_geometric.utils
import numpy as np
from torch.utils.data import SequentialSampler,RandomSampler

#Inputs for a custom loader
# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2,
#            persistent_workers=False)

import math
from typing import Union
from torch_geometric.data import Data
from torch_geometric.data import DataLoader, RandomNodeLoader
import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj, subgraph
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index

import ogb
import torch
import numpy as np
import os

from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.loader import DataLoader


# DATAPATH = 'data/splits/'

# def even_quantile_labels(vals, nclasses, verbose=True):
#     """ partitions vals into nclasses by a quantile based split,
#     where the first class is less than the 1/nclasses quantile,
#     second class is less than the 2/nclasses quantile, and so on
    
#     vals is np array
#     returns an np array of int class labels
#     """
#     label = -1 * np.ones(vals.shape[0], dtype=int)
#     interval_lst = []
#     lower = -np.inf
#     for k in range(nclasses - 1):
#         upper = np.nanquantile(vals, (k + 1) / nclasses)
#         interval_lst.append((lower, upper))
#         inds = (vals >= lower) * (vals < upper)
#         label[inds] = k
#         lower = upper
#     label[vals >= lower] = nclasses - 1
#     interval_lst.append((lower, np.inf))
#     if verbose:
#         print('Class Label Intervals:')
#         for class_idx, interval in enumerate(interval_lst):
#             print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
#     return label
# def get_adj_matrix(data):
#         adj_matrix = torch.zeros(data.num_nodes, data.num_nodes)
#         # edge_index stores edge list
#         # shape (2, |E|) where |E| is the number of edges in the graph

#         for e in range(data.num_edges):
#             src = data.edge_index[0][e]
#             tgt = data.edge_index[1][e]
#             adj_matrix[src][tgt] = 1
#         return adj_matrix


# def load_arxiv_year_dataset(root):
#     ogb_dataset = NodePropPredDataset(name='ogbn-arxiv',root=root)
#     graph = ogb_dataset.graph
#     graph['edge_index'] = torch.as_tensor(graph['edge_index'])
#     graph['node_feat'] = torch.as_tensor(graph['node_feat'])

#     label = even_quantile_labels(graph['node_year'].flatten(), 5, verbose=False)
#     label = torch.as_tensor(label).reshape(-1, 1)
#     # split_idx_lst = load_fixed_splits("arxiv-year",os.path.join(root,"splits"))

#     # train_mask = torch.stack([split["train"] for split in split_idx_lst],dim=1)
#     # val_mask = torch.stack([split["valid"] for split in split_idx_lst],dim=1)
#     # test_mask = torch.stack([split["test"] for split in split_idx_lst],dim=1)
#     data = Data(x=graph["node_feat"],y=torch.squeeze(label.long()),edge_index=graph["edge_index"])
#     return data


# def load_fixed_splits(dataset,split_dir):
#     """ loads saved fixed splits for dataset
#     """
#     name = dataset
#     splits_lst = np.load(os.path.join(split_dir,"{}-splits.npy".format(name)), allow_pickle=True)
#     for i in range(len(splits_lst)):
#         for key in splits_lst[i]:
#             if not torch.is_tensor(splits_lst[i][key]):
#                 splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
#     return splits_lst


# def mask_generation(index, num_nodes):
#         mask = torch.zeros(num_nodes, dtype = torch.bool)
#         # print(len(index))
#         mask[index] = 1
#         return mask


import math
from typing import Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index


class RandomNodeLoader1(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.RandomNodeLoader`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_parts (int): The number of partitions.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        **kwargs,
    ):
        self.data = data
        self.num_parts = num_parts

        if isinstance(data, HeteroData):
            edge_index, node_dict, edge_dict = to_homogeneous_edge_index(data)
            self.node_dict, self.edge_dict = node_dict, edge_dict
        else:
            edge_index = data.edge_index

        self.edge_index = edge_index
        self.num_nodes = data.num_nodes

        super().__init__(
            range(self.num_nodes),
            batch_size=math.ceil(self.num_nodes / num_parts),
            collate_fn=self.collate_fn,
            drop_last =True,
            **kwargs,
        )

    def collate_fn(self, index):
        if not isinstance(index, Tensor):
            index = torch.tensor(index)

        if isinstance(self.data, Data):
            return self.data.subgraph(index)

        elif isinstance(self.data, HeteroData):
            node_dict = {
                key: index[(index >= start) & (index < end)] - start
                for key, (start, end) in self.node_dict.items()
            }
            return self.data.subgraph(node_dict)

# Sample code to use FixedSizeNodeLoader
# data = load_arxiv_year_dataset('data/splits')
# splits = 5
# split_idx_lst = load_fixed_splits("arxiv-year", os.path.join("data/splits"))
# for split in range(splits):
#     split_idx = split_idx_lst[split]
#     train_idx = split_idx['train']
#     valid_idx = split_idx['valid']
#     test_idx = split_idx['test']
#     train_mask = mask_generation(train_idx, data.num_nodes)
#     valid_mask = mask_generation(valid_idx, data.num_nodes)
#     test_mask = mask_generation(test_idx, data.num_nodes)
#     data_pass = Data(x=data.x, edge_index=data.edge_index, num_classes=max(data.y).item() + 1, num_features=data.x.shape[1], y=data.y, train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
#     # loader = FixedSizeNodeLoader(data_pass, num_parts=10, batch_size=16934)
#     loader = RandomNodeLoader1(data_pass, num_parts=10)
#     for idx, data1 in enumerate(loader):
#         #data1 = collate_fn(data_pass,data1)
#         adj_t1 = get_adj_matrix(data1)
#         adj_t2 = to_dense_adj(data1.edge_index,max_num_nodes = data1.num_nodes)[0]
#         print(torch.all(adj_t2.eq(adj_t1)))
#         #if torch.eq(adj_t1,adj_t2):
#             #print("Problem Solved")
#         print(f"For {idx + 1} batch the shape of adj matrix is {adj_t1.shape}")


