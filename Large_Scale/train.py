import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
import os
import torch
import numpy as np
from utils import UtilFunctions
#from utils import UtilFunctions as utils
from test import ModelEvaluation


class ModelTraining():

    def __init__(self):

        return

    def train(self, model, train_loader, val_loader, data_obj, data_obj_name, train_iter, opti, hidden_layers, device, is_splits, split):
        torch.cuda.empty_cache()
        if not is_splits:
            model_path = os.getcwd() + "/saved_models/arxiv" + data_obj_name + "_" + model.name + "_" + f"split_{split}"+ "_" + model.activation_name+"_"+ str(model.dropout)+ "_"+str(model.nhid)+ "_"+str(hidden_layers) + "_layers_.pt"
            losses = []
            best_acc = 0.0
            # training loop
            for epoch in range(train_iter):
                model.train()
                opti.zero_grad()

                total_loss = total_examples = 0
                
                for idx, data in enumerate(train_loader):
                    data.x = data.x.to(device)
                    data.y = data.y.to(device) 
                    data.edge_index = data.edge_index.to(device)
                    adj_matrix = UtilFunctions().adj_generation(data.edge_index, data.num_nodes, data.num_edges).to(device)
                    # if data.edge_weight!= None:
                    #     data.edge_weight = data.edge_weight.to(device) 
                    if model.name =='GCNII':
                        pred = model(data)
                    else:
                        pred = model(data.x, adj_matrix)
                    label = data.y
                    pred = pred[data.train_mask]
                    label = label[data.train_mask]
                    loss = UtilFunctions.loss_fn(pred, label)
                    loss.backward()
                    total_loss += loss.item()
                    del data.edge_index,data.x,data.y,pred, label,loss,adj_matrix
                    torch.cuda.empty_cache()
                # total_loss.backward() 
                opti.step()
                    
                
                with torch.no_grad():
                    losses.append(total_loss)
                    total_test_acc = 0
                    for idx, data in enumerate(val_loader):
                        test_acc = ModelEvaluation().test(model, data, hidden_layers, model_path, device, is_validation = True,is_splits =is_splits,split=split)
                        total_test_acc+= test_acc
                    total_test_acc = total_test_acc/(len(val_loader))
                    print(f"Epoch: {epoch + 1:03d}, Loss: {total_loss:.4f}, Val_acc: {total_test_acc:.4f}")

                if total_test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), model_path)
                del total_loss
                torch.cuda.empty_cache()
            return model_path
        
        else:
            model_path = os.getcwd() + "/saved_models/" + data_obj.name.lower() + "_" + model.name + "_" + f"split_{split}"+ "_" + model.activation_name+"_"+ str(model.dropout)+ "_"+str(model.nhid)+ "_"+str(hidden_layers) + "_layers_.pt"
            losses = []
            best_acc = 0.0
            # training loop
            for epoch in range(train_iter):
                model.train()
                opti.zero_grad()

                total_loss = total_examples = 0
                
                for data in loader:
                    #data = data
                    print(data)
                    idx = 1 
                    print(f"Training for batch_{idx+1} and epoch_{epoch+1}") 
                    data = data.to(device)
                #emb, pred = model(data_obj.node_features, adj)
                    pred = model(data_obj.node_features, adj)
                    label = data_obj.node_labels
                    pred = pred[data_obj.train_mask[:,split]]
                    label = label[data_obj.train_mask[:,split]]
                    #pred = pred.to(device)
                    #label = label.to(device)
                    loss = UtilFunctions.loss_fn(pred, label)
                    loss.backward()
                    opti.step()
                    total_loss += loss.item() * data.num_nodes
                    total_examples += data.num_nodes
                    loss = total_loss / total_examples
                    print(f"Loss for batch_{idx+1} and epoch_{epoch+1}: {loss}")
                losses.append(loss)

                test_acc = ModelEvaluation().test(model, data_obj, adj, hidden_layers, model_path, device, is_validation = True,is_splits =is_splits,split=split)
                print(f"Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Val_acc: {test_acc:.4f}")

                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), model_path)

            return model_path