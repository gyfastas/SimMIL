import torch
import torch.nn as nn
from GraphBuilder.KNNGraphBuilder import KNNGraphBuilder
from GCN.GCNConv import GCNConv
from Aggregator.GraphWeightedAvgPooling import GraphWeightedAvgPooling
from torch_geometric.utils import dense_to_sparse

class GraphMILNet(nn.Module):
    """
    Notes:
        1. Data flow: (B*D1) -> GraphBuilder -> (B*D1, B*B) -> GCN -> (B*D2, B*B) -> Aggregator -> (1*D2, B*B)
        2. For default, the features should be normalized before this module.
    """
    def __init__(self, graph_builder, gcn, aggregator):
        super(GraphMILNet, self).__init__()
        self.graph_builder = graph_builder
        self.gcn = gcn
        self.aggregator = aggregator
    
    def forward(self, x):
        x, A = self.graph_builder(x)
        edge_index = dense_to_sparse(A)[0] #[2, E]
        x = self.gcn(x, edge_index)
        return self.aggregator(x, A)


if __name__ == "__main__":
    ## make configs
    K = 2
    x = torch.randn([128, 2048])
    graph_builder = KNNGraphBuilder(K)
    gcn = GCNConv(2048, 128)
    aggregator = GraphWeightedAvgPooling()
    model = GraphMILNet(graph_builder, gcn, aggregator)
    returns = model(x)
    print(returns.shape)
