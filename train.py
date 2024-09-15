import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import math
import os
import torch

from utils import UtilFunctions
from test import ModelEvaluation


class ModelTraining():

    def __init__(self):

        return

    def train(self, model, data_obj, adj, train_iter, opti, hidden_layers, device, is_splits, split):
        if not is_splits:
            model_path = os.getcwd() + "/saved_models/" + data_obj.name.lower() + "_" + model.name + "_" + model.activation_name+"_"+ str(model.dropout)+ "_"+str(model.nhid)+ "_"+str(hidden_layers) + "_layers_.pt"
            losses = []
            best_acc = 0.0



            # training loop
            for epoch in range(train_iter):

                model.train()
                opti.zero_grad()
                pred = model(data_obj.node_features, adj)
                label = data_obj.node_labels
                pred = pred[data_obj.train_mask]
                label = label[data_obj.train_mask]
                pred = pred.to(device)
                label = label.to(device)
                loss = UtilFunctions.loss_fn(pred, label)
                loss.backward()
                opti.step()

                losses.append(loss)

                test_acc = ModelEvaluation().test(model, data_obj, adj, hidden_layers, model_path, device, is_validation = True,is_splits=False,split = 0)
                print(f"Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val_acc: {test_acc:.4f}")

                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), model_path)

            return model_path
        
        else:
            model_path = os.getcwd() + "/saved_models/wSVD/" + data_obj.name.lower() + "_" + model.name + "_" + f"split_{split}"+ "_" + model.activation_name+"_"+ str(model.dropout)+ "_"+str(model.nhid)+ "_"+str(hidden_layers) + "_layers_.pt"
            losses = []
            best_acc = 0.0
            norms = []
            svds = []

            # training loop
            for epoch in range(train_iter):
                best_epoch = 0
                model.train()
                opti.zero_grad()
                
                torch.cuda.empty_cache()
                pred = model(data_obj.node_features, adj)
                label = data_obj.node_labels
                pred = pred[data_obj.train_mask[:,split]]
                label = label[data_obj.train_mask[:,split]]
                pred = pred.to(device)
                label = label.to(device)
                loss = UtilFunctions.loss_fn(pred, label)
                del pred,label
                torch.cuda.empty_cache()
                loss.backward()
                
               
                opti.step()

                losses.append(loss)
                

                test_acc = ModelEvaluation().test(model, data_obj, adj, hidden_layers, model_path, device, is_validation=True, is_splits=is_splits, split=split)
                print(f"Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val_acc: {test_acc:.4f}")

                if test_acc > best_acc:
                    best_epoch = epoch
                    best_acc = test_acc
                    torch.save(model.state_dict(), model_path)
                del test_acc,loss
                torch.cuda.empty_cache()


            return model_path
