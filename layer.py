import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class SANBet_Layer_Init(Module):
    """
    First MLP of SAN-Bet
    """

    def __init__(self, input_features, hidden_features, dropout=0.5):
        super(SANBet_Layer_Init, self).__init__()
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(input_features, hidden_features)
        self.linear2 = torch.nn.Linear(hidden_features, hidden_features)
        self.linear3 = torch.nn.Linear(hidden_features,1)


    def forward(self, adj):
        x = F.relu(self.linear1(adj))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.linear3(x))
        x = F.dropout(x, self.dropout, self.training)
        return x

class SANBet_Layer(Module):
    """
    Message Passing Layer
    """

    def __init__(self, bias=True):
        super(SANBet_Layer, self).__init__()
        self.weight = Parameter(torch.zeros(1))
        if bias:
            self.bias = Parameter(torch.ones(1))

    def forward(self, inp, adj):
        support = inp * self.weight
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output