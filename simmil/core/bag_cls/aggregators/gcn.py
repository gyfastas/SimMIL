import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, GlobalAttention
from torch_geometric.utils import dense_to_sparse

class GCNPooling(nn.Module):
    """
    GCN Pooling with: 
        (a) distance based Graph Building (threshold based, thres = 0.5 * max(dist))
        (b) three Chebshev conv (without changing feat dim)
        (c) graph self attention pooling (global pooling)

    Args:
        in_feat: (int) input feature dimension
        attention_feat: (int) attention hidden feature dimension

    """
    def __init__(self, in_feat=2048, attention_feat=256):
        super().__init__()
        self.in_feat = in_feat
        self.attention_feat = attention_feat
        self.conv1 = ChebConv(in_feat, in_feat, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = ChebConv(in_feat, in_feat, 3)
        self.relu2 = nn.ReLU()
        self.conv3 = ChebConv(in_feat, in_feat, 3)
        self.relu3 = nn.ReLU()
        self.gate_nn = nn.Sequential(
            nn.Linear(self.in_feat, self.attention_feat),
            nn.Tanh(),
            nn.Linear(self.attention_feat, 1)
        )
        self.pooling = GlobalAttention(gate_nn=self.gate_nn)

    def build_graph(self, feats, batch=None):
        dot_product = torch.matmul(feats, feats.T)# [N, D] => [N, N]
        l2_sqaure = (feats**2).sum(1)
        norm_matrix = l2_sqaure.view(-1, 1).repeat([1, l2_sqaure.shape[0]]) + l2_sqaure.view(1, -1).repeat([l2_sqaure.shape[0], 1])
        dist_matrix = torch.sqrt((norm_matrix - 2 * dot_product))

        adj_matrix = (dist_matrix < (0.5 * dist_matrix.max()))
        if batch is not None:
            not_same_batch = batch.view(-1, 1).eq(batch.view(1, -1))
            adj_matrix[not_same_batch] = 0.0
        edge_index, edge_value = dense_to_sparse(adj_matrix)
        return feats, edge_index

    def graph_conv(self, feats, edge_index, batch=None):
        feats = self.conv1(feats, edge_index, batch=batch)
        feats = self.relu1(feats)
        feats = self.conv2(feats, edge_index, batch=batch)
        feats = self.relu2(feats)
        feats = self.conv3(feats, edge_index, batch=batch)
        feats = self.relu3(feats)
        return feats, edge_index

    def graph_pooling(self, feats, batch=None):
        return self.pooling(feats, batch=batch)

    def forward(self, feats):
        batch_size, bag_size, in_feat = feats.shape
        batch = torch.arange(0, batch_size).view(-1, 1).repeat([bag_size, 1]).flatten().to(feats.device)
        feats = feats.view(batch_size*bag_size, in_feat) # [N, B, D] => [N*B, D]
        feats, edge_index = self.build_graph(feats, batch)
        feats, edge_index = self.graph_conv(feats, edge_index, batch)
        feats = self.graph_pooling(feats, batch=batch)
        return feats.view(batch_size, in_feat)


def gcn():
    return GCNPooling()