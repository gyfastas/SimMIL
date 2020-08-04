import torch
import torch.nn as nn
class GraphCon(nn.Module):
    def __init__(self, base_encoder, base_agg, arch, pretrained, m, attention, graph, n_topk):
        super(GraphCon, self).__init__()
        self.encoder_q = base_encoder(arch, pretrained)
        self.encoder_k = base_encoder(arch, pretrained)
        self.agg_q = base_agg(attention, graph, n_topk)
        self.agg_k = base_agg(attention, graph, n_topk)
        self.m = m

    def _momentum_update_key_backbone_and_agg(self):
        """
               Momentum update of the key backbone&agg
               """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.agg_q.parameters(), self.agg_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def forward(self, im_q, im_k, batch):
        fea_q, self_feat_q = self.encoder_q(im_q)
        Y_prob_q, Affinity_q, G_q = self.agg_q(fea_q, batch)
        if im_k is None:
            return Y_prob_q, None, None, None
        with torch.no_grad():
            self._momentum_update_key_backbone_and_agg()
            fea_k, self_feat_k = self.encoder_k(im_k)
            _, Affinity_k, G_k = self.agg_q(fea_k, batch)

        return Y_prob_q, (Affinity_q, Affinity_k), (G_q, G_k), (self_feat_q, self_feat_k)


if __name__ =='__main__':
    from Builder.FeaAgg import ResBackbone
    from Builder.FeaAgg import Agg_Residual
    model = GraphCon(ResBackbone, Agg_Residual, 'resnet18', True, 0.9)
    x_1 = torch.randn([3, 3, 224, 224])
    x_2 = torch.randn([3, 3, 224, 224])
    label = torch.tensor([0])
    batch = torch.tensor([0,0,0])

    out = model(x_1, x_2, batch)
    pass
