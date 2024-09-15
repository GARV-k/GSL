import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import math
from utils import UtilFunctions


class ModelEvaluation():

    def __init__(self):

        return

    def test(self, model, data_obj, hidden_layers, model_path, device, is_validation,is_splits, split):
        
        torch.cuda.empty_cache()
        data_obj.x = data_obj.x.to(device) 
        data_obj.edge_index = data_obj.edge_index.to(device)
        adj_matrix = UtilFunctions().adj_generation(data_obj.edge_index, data_obj.num_nodes, data_obj.num_edges).to(device)
        
        if is_validation is False:
        
            model.load_state_dict(torch.load(model_path))

            model.eval()
            correct = 0
            if model.name == 'GCNII':
                pred = model(data_obj)
            else:
                pred = model(data_obj.x, adj_matrix)
            pred = pred.argmax(dim = 1)
            label = data_obj.y
            pred = pred[data_obj.test_mask]
            label = label[data_obj.test_mask]
            pred = pred.to(device)
            label = label.to(device)
            correct = pred.eq(label).sum().item()
            accuracy = correct / int(data_obj.test_mask.sum())
            del pred,label,correct,adj_matrix,data_obj.x,data_obj.edge_index 
            torch.cuda.empty_cache()
            return accuracy 
        else: 
            model.eval()
            correct = 0
            if model.name == 'GCNII':
                pred = model(data_obj)
            else:
                pred = model(data_obj.x, adj_matrix)
            pred = pred.argmax(dim = 1)
            label = data_obj.y
            pred = pred[data_obj.valid_mask]
            label = label[data_obj.valid_mask]
            pred = pred.to(device)
            label = label.to(device)
            correct = pred.eq(label).sum().item()
            accuracy = correct / int(data_obj.valid_mask.sum())
            del pred,label,correct,adj_matrix,data_obj.x,data_obj.edge_index 
            torch.cuda.empty_cache()
            return accuracy
