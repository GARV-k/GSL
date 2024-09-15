import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import os 

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from cornell_loader import Cornell
from texas_loader import Texas
from wisconsin_loader import Wisconsin
from actor_loader import Actor_new
from chameleon_loader import Chameleon
from squirrel_loader import Squirrel
#from torch_geometric.datasets import Actor
#from coauthor_cs_loader import CoauthorCS

from train import ModelTraining
from test import ModelEvaluation
from utils import UtilFunctions
import pickle
from sklearn.manifold import TSNE
from model import GCN_L,GCN,GCN2
import argparse

def std_dev(test_list):
    mean = sum(test_list) / len(test_list) 
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list) 
    res = variance ** 0.5
    return res

def argument_parser():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type_GCN', help = 'type of GCN used', default = 'normal', type = str)
    parser.add_argument('--dataset', help = 'enter name of dataset in smallcase', default = 'cora', type = str)
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
    parser.add_argument('--alpha', help = 'slope of leaky relu', default = 1, type = float)
    parser.add_argument('--theta', help = 'slope of leaky relu', default = 1, type = float)

    parser.add_argument('--dropout', help = 'Dropoout in the layers', default = 0.60, type = float)
    parser.add_argument('--w_decay', help = 'Weight decay for the optimizer', default = 0.0005, type = float)
    parser.add_argument('--device', help = 'cpu or gpu device to be used', default = 'cpu', type = None)

    return parser

parsed_args = argument_parser().parse_args()
type_GCN = parsed_args.type_GCN
dataset = parsed_args.dataset
activation = parsed_args.activation
splits = parsed_args.splits
lr = parsed_args.lr
seed = parsed_args.seed
hidden_layers = parsed_args.hidden_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
use_saved_model = parsed_args.use_saved_model
alpha = parsed_args.alpha
theta = parsed_args.theta
dropout = parsed_args.dropout
weight_decay = parsed_args.w_decay
device = parsed_args.device
print("All the Params are as follows: ")
print("Device: ", device)
print(f"learning Rate : {lr}")
print(f"nhid : {hidden_dim}")
print(f"dropout : {dropout}")
print(f"activation : {activation}")
print(f"alpha,theta : {alpha},{theta}")
print(f"wdecay : {weight_decay}")
print(f"epochs : {train_iter}")


if dataset == 'cornell':

    data_obj = Cornell()

elif dataset == 'texas':

    data_obj = Texas()

elif dataset == 'wisconsin':

    data_obj = Wisconsin()
    
elif dataset == 'chameleon':

    data_obj = Chameleon()
    
elif dataset == 'squirrel':
    data_obj = Squirrel()

elif dataset == 'actor':
    data_obj = Actor_new()

else:

    print("Incorrect name of dataset")

data_obj.node_features = data_obj.node_features.to(device)
data_obj.node_labels = data_obj.node_labels.to(device)

if splits == 1:
            
        adj_matrix = UtilFunctions().adj_generation(data_obj.edge_index, data_obj.num_nodes, data_obj.num_edges).to(device)
        
        if type_GCN == 'structured':
            model = GCN_L(activation, hidden_layers, data_obj.num_nodes, data_obj.num_features, hidden_dim, data_obj.num_classes, dropout, device, True)
        elif type_GCN == 'normal':
            model = GCN(activation, hidden_layers, data_obj.num_nodes, data_obj.num_features, hidden_dim, data_obj.num_classes, dropout, device, True)
        elif type_GCN == '2_normal':
            model = GCN2(activation, hidden_layers, data_obj.num_nodes, data_obj.num_features, hidden_dim, data_obj.num_classes, dropout, device, True,1,theta = 1,shared_weights=True)
        opti = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        model = model.to(device)

        # print("Model Name: RPE-GNN")
        print("Model_GCN:", type_GCN)
        print("Dataset:", dataset.upper())
        print("Hidden Layers:", hidden_layers)

        if use_saved_model == 'False':

            # training of the model
            print("Optimization started....")
            trainer = ModelTraining()
            model_path = trainer.train(model, data_obj, adj_matrix, train_iter, opti, hidden_layers, device, is_splits=False,split = 0)

        else:

            print("Trained model loaded from the directory...")
            model_path = os.getcwd() + "/saved_models/" + data_obj.name.lower() + "_" + model.name + "_" + str(hidden_layers) + "_layers_.pt"


        # evaluation
        print("Evaluating on Test set")
        avg_acc = 0.0
        max_acc = 0.0
        eval = ModelEvaluation()
        acc_list = []

        for i in range(test_iter):

            acc = eval.test(model, data_obj, adj_matrix, hidden_layers, model_path, device, is_validation = False,is_splits=False,split = 0)
            acc_list.append(acc)
            if acc > max_acc:
                max_acc = acc
            avg_acc += acc
            print("Test iteration:", i+1, " complete --- accuracy ",acc)

        avg_acc /= test_iter
        std_deviation = std_dev(acc_list)
        print(f'Maximum accuracy on Test set: {max_acc:.4f}')
        print(f'Average accuracy on Test set: {avg_acc:.4f}')
        print(f'Std dev on Test set: {std_deviation:.4f}')
        print(f'The list of acc on test set: {acc_list}')




else:
        

        if use_saved_model == 'False':
            splits = 3

            acc_list = []            
            # training of the model
            print("Optimization started....")
            trainer = ModelTraining()
            avg_acc = 0.0
            max_acc = 0.0
            for split in range(splits):
                
                adj_matrix = UtilFunctions().adj_generation(data_obj.edge_index, data_obj.num_nodes, data_obj.num_edges).to(device)
                
                if type_GCN == 'structured':
                    model = GCN_L(activation, hidden_layers, data_obj.num_nodes, data_obj.num_features, hidden_dim, data_obj.num_classes, dropout, device, True)
                elif type_GCN == 'normal':
                    model = GCN(activation, hidden_layers, data_obj.num_nodes, data_obj.num_features, hidden_dim, data_obj.num_classes, dropout, device, True)
                elif type_GCN == '2_normal':
                    model = GCN2(activation, hidden_layers, data_obj.num_nodes, data_obj.num_features, hidden_dim, data_obj.num_classes, dropout, device, True,alpha,theta,shared_weights=True)
                opti = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
                model = model.to(device)
        
                print("Model_GCN:", type_GCN)
                print("Dataset:", dataset.upper())
                print("Hidden Layers:", hidden_layers)
                print(f"Training on Train set for split {split+1}")
                model_path = trainer.train(model, data_obj, adj_matrix, train_iter, opti, hidden_layers, device,True,split)
                
                print(f"Evaluating on Test set for split {split+1}")
                
                eval = ModelEvaluation()

                acc = eval.test(model, data_obj, adj_matrix, hidden_layers, model_path, device, is_validation = False, is_splits=True, split=split)
                acc_list.append(acc)
                if acc > max_acc:
                    max_acc = acc
                avg_acc += acc
                print("Test iteration for split", split+1, " complete --- accuracy ",acc)

            avg_acc /= splits
            std_deviation = std_dev(acc_list)
            print(f'Maximum accuracy on Test set for all splits: {max_acc:.4f}')
            print(f'Average accuracy on Test set for all splits: {avg_acc:.4f}')
            print(f'Std dev on Test set for all splits: {std_deviation:.4f}')
            print(f'The list of acc on test set for all splits: {acc_list}')

        else:
            #This is buggy, havent incorporated the splits thing here, so better to not try to access this
            print("Trained model loaded from the directory...")
            model_path = os.getcwd() + "/saved_models/" + data_obj.name.lower() + "_" + model.name + "_" + str(hidden_layers) + "_layers_.pt"
