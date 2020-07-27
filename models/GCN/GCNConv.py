import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # "Add" aggregation (Step 5).
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from GraphBuilder.KNNGraphBuilder import KNNGraphBuilder
    from torch_geometric.utils import dense_to_sparse
    x = nn.functional.normalize(torch.randn([4, 128]))
    builder = KNNGraphBuilder(2)
    gcn = GCNConv(128, 32)  
    x, adjacent = builder(x)
    edge_index = dense_to_sparse(adjacent)[0]
    print("adjacent matrix: ", adjacent)
    print("edges: ", edge_index)
    z = gcn(x, edge_index)
    print(z.shape)
