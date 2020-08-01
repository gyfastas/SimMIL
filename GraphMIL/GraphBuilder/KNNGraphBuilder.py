import torch
import numpy as np
import torch.nn as nn

def topk_to_adjacent(sim, idxs):
    """
    Convert TopK style indexs to adjacent matrix (binary mask)
    Args:
        idxs: (torch.Tensor) [B, K]
    Returns:
        adjacent_matrix: (torch.Tensor) [B, B] the weight for each edge is the similarity between instances
    """
    raw_idxs = torch.arange(0, idxs.shape[0]).to(idxs.device)
    raw_idxs = raw_idxs.view(idxs.shape[0], 1)
    raw_idxs = raw_idxs.repeat(1, idxs.shape[1])
    T_index = torch.stack([raw_idxs, idxs]).view(2, idxs.shape[0] * idxs.shape[1])
    adjacent_matrix = torch.zeros([idxs.shape[0], idxs.shape[0]]).to(idxs.device)
    adjacent_matrix[T_index[0, :], T_index[1, :]] = sim[T_index[0, :], T_index[1, :]]
    return adjacent_matrix 

class KNNGraphBuilder(nn.Module):
    def __init__(self, K):
        super(KNNGraphBuilder, self).__init__()
        self.K = K

    def forward(self, x):
        similarity_matrix = torch.matmul(x, x.T) # (B*D, D*B) -> (B, B)
        ## Build adjacent matrix
        _, idxs = similarity_matrix.topk(self.K)
        adjacent_matrix = topk_to_adjacent(similarity_matrix, idxs) #(B, K) -> (B, B) Adjacent Matrix
        return x, adjacent_matrix

if __name__=="__main__":
    x = nn.functional.normalize(torch.randn([4, 128]))
    builder = KNNGraphBuilder(2)
    print(builder(x)[1])
        
