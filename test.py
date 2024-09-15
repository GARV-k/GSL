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

    def test(self, model, data_obj, adj, hidden_layers, model_path, device, is_validation,is_splits, split):
        
        if not is_splits:
            if is_validation is False:
            
                model.load_state_dict(torch.load(model_path))

            model.eval()
            correct = 0
            pred = model(data_obj.node_features, adj)
            pred = pred.argmax(dim = 1)
            label = data_obj.node_labels
            pred = pred[data_obj.test_mask]
            label = label[data_obj.test_mask]
            pred = pred.to(device)
            label = label.to(device)
            correct = pred.eq(label).sum().item()
            accuracy = correct / int(data_obj.test_mask.sum())
            
            return accuracy 
        
        else:
            if is_validation is False:
            
                model.load_state_dict(torch.load(model_path))
                
                model.eval()
                correct = 0
                
                pred = model(data_obj.node_features, adj)
                pred = pred.argmax(dim = 1)
                label = data_obj.node_labels
                pred = pred[data_obj.test_mask[:,split]]
                label = label[data_obj.test_mask[:,split]]
                
                correct = pred.eq(label).sum().item()
                accuracy = correct / int(data_obj.test_mask[:,split].sum())
                
                return accuracy 
               
            
            else:
                model.eval()
                correct = 0
               
                pred = model(data_obj.node_features, adj)
                pred = pred.argmax(dim = 1)
                label = data_obj.node_labels
                pred = pred[data_obj.val_mask[:,split]]
                label = label[data_obj.val_mask[:,split]]
               
                correct = pred.eq(label).sum().item()
                accuracy = correct / int(data_obj.val_mask[:,split].sum())
                
                
                return accuracy