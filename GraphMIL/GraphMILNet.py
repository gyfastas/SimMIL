import torch
import torch.nn as nn
from GraphBuilder.KNNGraphBuilder import KNNGraphBuilder
from GCN.GCNConv import GCNConv
from Aggregator.GraphWeightedAvgPooling import GraphWeightedAvgPooling
from torch_geometric.nn import SAGPooling, GlobalAttention
from torch_geometric.utils import dense_to_sparse, to_dense_adj

class GraphMILNet(nn.Module):
    """
    Notes:
        1. Data flow: (B*D1) -> GraphBuilder -> (B*D1, B*B) -> GCN -> (B*D2, B*B) -> Aggregator -> (1*D2, B*B)
        2. For default, the features should be normalized before this module.
    """
    def __init__(self, graph_builder, gcn, aggregator, dense_aggregation=False):
        super(GraphMILNet, self).__init__()
        self.graph_builder = graph_builder
        self.gcn = gcn
        self.aggregator = aggregator
        self.dense_aggregation = dense_aggregation
    
    def forward(self, x):
        x, A = self.graph_builder(x)
        edge_index = dense_to_sparse(A)[0] #[2, E]
        x = self.gcn(x, edge_index)
        if self.dense_aggregation:
            return self.aggregator(x, A)
        else:
            return self.aggregator(x, edge_index)

def base_net(K=2, in_channels=2048, out_channels=128):
    graph_builder = KNNGraphBuilder(K)
    gcn = GCNConv(2048, 128)
    aggregator = GraphWeightedAvgPooling()
    net = GraphMILNet(graph_builder, gcn, aggregator, True)
    return net

def single_layer_gcn_with_sag_pooling(K=2, in_channels=2048, out_channels=128):
    graph_builder = KNNGraphBuilder(K)
    gcn = GCNConv(in_channels, out_channels)
    ## could be modified
    sag_pooling_ratios  = 0.5
    gate_nn = nn.Sequential(nn.Linear(int(out_channels * sag_pooling_ratios), 1))
    class SAGGlobalPooling(nn.Module):
        def __init__(self, gate_nn, in_channels=128, sag_pooling_ratios=0.5):
            super(SAGGlobalPooling, self).__init__()
            self.sag_pooling = SAGPooling(in_channels, sag_pooling_ratios)
            self.global_pooling = GraphWeightedAvgPooling()
        
        def forward(self, x, A):
            edge_index, edge_attr = dense_to_sparse(A)
            x, edge_index, edge_attr, _, _, _ = self.sag_pooling(x, edge_index, edge_attr)
            new_A = to_dense_adj(edge_index, edge_attr = edge_attr)
            out = self.global_pooling(x, new_A)
            return out
        
    aggregator = SAGGlobalPooling(gate_nn, out_channels, sag_pooling_ratios)
    net = GraphMILNet(graph_builder, gcn, aggregator, True)
    return net

if __name__ == "__main__":
    ## make configs
    K = 2
    x = torch.randn([128, 2048])
    model = single_layer_gcn_with_sag_pooling(K, 2048, 128)
    returns = model(x)
    print(returns.shape)
    print(model)
