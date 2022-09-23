import torch.nn as nn


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats):
        # feats: [N, B, D]
        return feats.mean(1)

def mean_pooling():
    return MeanPooling()