import torch
import torch.nn as nn


class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats):
        # feats: [N, B, D]
        return torch.max(feats, dim=1)[0]

def max_pooling():
    return MaxPooling()