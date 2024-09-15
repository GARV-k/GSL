import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import os 
from torch_geometric.datasets import Flickr
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils
from torch_geometric.loader import GraphSAINTRandomWalkSampler,ShaDowKHopSampler,RandomNodeLoader
import matplotlib   .pyplot as plt
from arxiv_year_loader import load_arxiv_year_dataset, load_fixed_splits, even_quantile_labels
import os.path as osp
from genius_loader import *
from Penn94_loader import *
from pokec_loader import load_pokec_mat 
from pokec_loader import load_fixed_splits as splits_load
from genius_loader import load_genius
from twitch_gamers_loader import load_twitch_gamer_dataset
from snap_patents_loader import load_snap_patents_mat
from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj
from train import ModelTraining
from test import ModelEvaluation
from utils import UtilFunctions as utils
from fixed_size_NodeLoader import RandomNodeLoader1
from sklearn.manifold import TSNE
from model2 import GCN_L,GCN,GCNII
import argparse
import math

def std_dev(test_list):
    mean = sum(test_list) / len(test_list) 
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list) 
    res = variance ** 0.5
    return res

def argument_parser():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type_GCN', help = 'type of GCN used', default = 'normal', type = str)
    parser.add_argument('--dataset', help = 'enter name of dataset in smallcase', default = 'cora', type = str)
    parser.add_argument('--loader_params', help='List of the params used in batch', default=10, type=int)
    parser.add_argument('--activation', help = 'enter name of activation used in Structured Learning', default = 'relu', type = str)
    parser.add_argument('--splits', help = 'no. of splits in the data', default = 1, type = int)
    parser.add_argument('--lr', help = 'learning rate', default = 0.2, type = float)
    parser.add_argument('--seed', help = 'Random seed', default = 100, type = int)
    parser.add_argument('--hidden_layers', help = 'number of hidden layers', default = 2, type = int)
    parser.add_argument('--hidden_dim', help = 'hidden dimension for node features', default = 16, type = int)
    parser.add_argument('--train_iter', help = 'number of training iteration', default = 100, type = int)
    parser.add_argument('--test_iter', help = 'number of test iterations', default = 1, type = int)
    parser.add_argument('--use_saved_model', help = 'use saved model in directory', default = False, type = None)
    #parser.add_argument('--nheads', help = 'Number of attention heads', default = False, type = int)
    parser.add_argument('--alpha', help = 'slope of leaky relu', default = False, type = float)
    parser.add_argument('--theta', help = 'slope of leaky relu', default = False, type = float)
    parser.add_argument('--dropout', help = 'Dropoout in the layers', default = 0.60, type = float)
    parser.add_argument('--w_decay', help = 'Weight decay for the optimizer', default = 0.0005, type = float)
    parser.add_argument('--device', help = 'cpu or gpu device to be used', default = 'cpu', type = None)

    return parser

parsed_args = argument_parser().parse_args()
type_GCN = parsed_args.type_GCN
dataset = parsed_args.dataset
loader_params = parsed_args.loader_params
activation = parsed_args.activation
splits = parsed_args.splits
lr = parsed_args.lr
seed = parsed_args.seed
hidden_layers = parsed_args.hidden_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
use_saved_model = parsed_args.use_saved_model
#nheads = parsed_args.nheads
alpha = parsed_args.alpha
theta = parsed_args.theta
dropout = parsed_args.dropout
weight_decay = parsed_args.w_decay

device = parsed_args.device
print("Device: ", device)

if dataset == 'arxiv':
    data = load_arxiv_year_dataset('data/splits')
    split_idx_lst = load_fixed_splits("arxiv-year",None)
                                      
elif dataset == 'penn94':
    data = load_fb100_dataset('Penn94')
    split_idx_lst = load_fixed_splits("fb100-Penn94",None) 
elif dataset == 'pokec':
    data = load_pokec_mat()
    split_idx_lst = splits_load("pokec",None) 
elif dataset == 'genius':
    data = load_genius()
    split_idx_lst = load_fixed_splits("genius",None) 
elif dataset == 'twitch':
    data = load_twitch_gamer_dataset()
    split_idx_lst = load_fixed_splits("twitch-gamers",None) 
elif dataset == 'snap':
    data = load_snap_patents_mat()
    split_idx_lst = load_fixed_splits("snap-patents",None) 
else:

    print("Incorrect name of dataset")

print("Optimization started....")
#print()
trainer = ModelTraining()
avg_acc = 0.0
max_acc = 0.0
acc_list = []
print(f"Number of splits : {len(split_idx_lst)}.")
for split in range(len(split_idx_lst)):
    print(f"for loop in split_{split+1 }:")
    split_idx = split_idx_lst[split]
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    print(data.num_nodes)
    train_mask = utils.mask_generation(train_idx, data.num_nodes)
    valid_mask = utils.mask_generation(valid_idx, data.num_nodes)
    test_mask = utils.mask_generation(test_idx, data.num_nodes)
    data_pass = Data(x=data.x, edge_index = data.edge_index, num_classes=max(data.y).item() + 1, num_features = data.x.shape[1], y=data.y, train_mask=train_mask, valid_mask =valid_mask,test_mask=test_mask)

    train_loader = RandomNodeLoader1(data_pass,loader_params)   
    val_loader = RandomNodeLoader1(data_pass,loader_params) 
    print(f'Length of Train loader: {len(train_loader)}')
    print(f'Length of Val loader: {len(val_loader)}')
    print(f"Total no. of nodes : {data_pass.num_nodes}")
    print(f"Total no. of nodes in each batch: {math.ceil(data_pass.num_nodes / loader_params)}")
    for data1 in train_loader:
        num_nodes = data1.num_nodes
        break
    if type_GCN == 'structured':
        model = GCN_L(activation, hidden_layers, num_nodes, data_pass.num_features, hidden_dim, data_pass.num_classes, dropout, device, True)
    elif type_GCN == 'normal':
        model = GCN(activation, hidden_layers, num_nodes, data_pass.num_features, hidden_dim, data_pass.num_classes, dropout, device, True)
    elif type_GCN == '2_normal':
        model = GCNII(activation, hidden_layers, num_nodes, data_pass.num_features, hidden_dim, data_pass.num_classes, dropout, device, True,alpha,theta,shared_weights=True)
        
    opti = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    model = model.to(device)
    print("Model_GCN:", type_GCN)
    print("Dataset:", dataset.upper())
    print("Hidden Layers:", hidden_layers)
    print(f"Training on Train set for split {split+1}")
    model_path = trainer.train(model,train_loader,val_loader,data_pass,dataset, train_iter, opti, hidden_layers, device,False,split)
    
    print(f"Evaluating on Test set for split {split+1}")
    eval = ModelEvaluation()
    
    test_loader = RandomNodeLoader1(data_pass,loader_params)

    #for i in range(test_iter):
    acc = 0
    
    for idx, data1 in enumerate(test_loader):
        if idx == 0:
            batch_acc = eval.test(model, data1,hidden_layers,model_path,device,is_validation = False, is_splits=False, split=split)
        print(f"acc for test set of batch {idx+1}: {batch_acc}  ")
        acc+=batch_acc
    acc = acc/(len(test_loader))
    print("Test iteration for split", split+1, " complete --- accuracy for all baches is :",acc)
    acc_list.append(acc)
    if acc > max_acc:
        max_acc = acc
    avg_acc += acc
avg_acc /= splits
std_deviation = std_dev(acc_list)
print(f'Maximum acc on Test set from all splits: {max_acc:.4f}')
print(f'Average accuracy on Test set for all splits: {avg_acc:.4f}')
print(f'Std dev on Test set for all splits: {std_deviation:.4f}')
print(f'The list of acc on test set for all splits: {acc_list}')


