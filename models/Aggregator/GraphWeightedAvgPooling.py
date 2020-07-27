import torch
import torch.nn as nn

def graph_weighted_avg_pooling(x, A):
    """
    pooling based on Adjacent Matrix as weights.
    We assume that node with higher degree is more "important"
    
    Args:
        x: (torch.tensor) [B, D]
        A: (torch.tensor) [B, B]
    Returns:
        torch.tensor [D]
    """
    weights = (A.sum(1) / A.sum()).view(x.shape[0], 1) # [B, 1]
    return (x*weights).sum(0)

class GraphWeightedAvgPooling(nn.Module):
    """
    Easy-implemented graph average pooling.
    Notes:
        1. The weight is the adjacent matrix, which could be defined as "similarity between instances"
    """
    def __init__(self):
        super(GraphWeightedAvgPooling, self).__init__()
    
    def forward(self, x, A):
        return graph_weighted_avg_pooling(x, A)

if __name__ == "__main__":
    pooling_layer = GraphWeightedAvgPooling()
    x = torch.nn.functional.normalize(torch.randn([4, 4]).sigmoid())
    A = torch.matmul(x, x.T)
    print("Adjacent: ", A)
    print("x:", x)
    print("pooling result: ", pooling_layer(x, A))