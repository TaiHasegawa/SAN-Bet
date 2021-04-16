
import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model import SANBet
torch.manual_seed(20)
import argparse


def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    model.train()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()

def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test):
    model.eval()
    list_topk = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)

        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)

        topk = topk_overlap_ratio(y_out, true_val, num_nodes, model_size, k_ratio)
        list_topk.append(topk)

    print(f"   Average top-{k_ratio*100}% overlap ratio on test graphs is: {np.mean(np.array(list_topk))} and std: {np.std(np.array(list_topk))}")


parser = argparse.ArgumentParser()
parser.add_argument("--g",default="SF")
parser.add_argument("--k", type=float, default=0.01)
args = parser.parse_args()
gtype = args.g 
k_ratio = args.k


# parameters
hidden = 10
num_epoch =  10
model_size = 10000

#Loading graph data
if gtype == "SF":
    data_path = "./datasets/data_splits/SF-"
    print("Scale-free graphs selected.")
elif gtype == "ER":
    data_path = "./datasets/data_splits/ER-"
    print("Erdos-Renyi random graphs selected.")
elif gtype == "GRP":
    data_path = "./datasets/data_splits/GRP-"
    print("Gaussian Random Partition graphs selected.")

#Load training data
print(f"Loading data...")
with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)

#Load test data
with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")
list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SANBet(input_features=model_size, hidden_features=hidden, dropout=0.6)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

print("Training")
print(f"Total Number of epoches: {num_epoch}")

for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)

    #to check test loss while training
    with torch.no_grad():
        test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)