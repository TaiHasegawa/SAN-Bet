import torch.nn as nn
import torch.nn.functional as F
from layer import SANBet_Layer
from layer import SANBet_Layer_Init
import torch 


class SANBet(nn.Module):
    def __init__(self, input_features, hidden_features, dropout):
        super(SANBet, self).__init__()

        self.gscn1 = SANBet_Layer_Init(input_features, hidden_features, dropout)
        self.gscn2 = SANBet_Layer()
        self.gscn3 = SANBet_Layer()


    def forward(self,adj1,adj2):

        x_1 = F.relu(self.gscn1(adj1))
        x2_1 = F.relu(self.gscn1(adj2))

        x_2 = F.relu(self.gscn2(x_1, adj1))
        x2_2 = F.relu(self.gscn2(x2_1, adj2))

        x_3 = F.relu(self.gscn3(x_2, adj1))
        x2_3 = F.relu(self.gscn3(x2_2, adj2))


        out1 = x_1 + x_2 + x_3
        out2 = x2_1 + x2_2 + x2_3

        x = torch.mul(out1, out2)

        return x