import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    in_feat: input feature dimension
    hidden_feat: hidden feature dimension
    head: head number, 1 as default (single-head attention)
    """
    def __init__(self, in_feat=512, hidden_feat=64, head=1):
        super().__init__()
        self.in_feat = in_feat
        self.hiddent_feat = hidden_feat
        self.head = head

        self.attention = nn.Sequential(
            nn.Linear(self.in_feat, self.hiddent_feat),
            nn.Tanh(),
            nn.Linear(self.hiddent_feat, self.head)
        )

    def forward(self, feats):
        """
        feats: torch.tensor([batch_size, bag_size, in_feat])
        """
        batch_size, bag_size, in_feat = feats.shape
        A = self.attention(feats.view(batch_size*bag_size, in_feat))  # (batch_size x bag_size, head)
        A = A.view(batch_size, bag_size, self.head).softmax(dim=1)
        return torch.bmm(A.permute(0,2,1), feats).view(batch_size, in_feat)

def attention():
    return AttentionPooling(2048, 256, 1)
