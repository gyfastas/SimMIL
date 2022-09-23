import torch.nn as nn
import torch


class BagMILModel(nn.Module):
    def __init__(self, backbone_factory, aggregator_factory, class_num, backbone_params=None, aggregator_params=None):
        super().__init__()
        self.backbone = backbone_factory(**backbone_params) if backbone_params is not None else backbone_factory()
        self.aggregator = aggregator_factory(**aggregator_params) if aggregator_params is not None else aggregator_factory()
        self.classifier = nn.Linear(self.backbone.fc.in_features, class_num)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()
        self.backbone.fc = nn.Flatten()

    def forward(self, x):
        # x: [N, B, C, H, W]
        n,b,_,_,_ = x.shape
        feats = self.backbone(x.view(n*b, *x.shape[2:])) # feats: [N*B, D]
        feats = feats.view(n, b, -1)
        feats = self.aggregator(feats) # [N, B, D] => [N, D]
        return self.classifier(feats)