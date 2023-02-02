import torch.nn.functional as F
import torch

from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, out_dim, cached=True,
                             normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        return x
