import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import Linear, SAGEConv, GATConv, TransformerConv, HeteroConv, to_hetero


class GAT(nn.Module):
    def __init__(self, hidden_dim, num_classes, heads=3, num_layers=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.heads = heads
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lins = nn.ModuleList()

        for i in range(self.num_layers):
            self.convs.append(GATConv((-1, -1), hidden_dim, heads=heads))
            self.lins.append(Linear(-1, hidden_dim * heads))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        self.fc_out = Linear(-1, num_classes)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) + self.lins[i]
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc_out(x)

        return x
