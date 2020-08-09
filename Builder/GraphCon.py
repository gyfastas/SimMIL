import torch
import torch.nn as nn
class GraphCon(nn.Module):
    def __init__(self, mode, base_encoder, arch, pretrained,
                 base_agg, attention, graph, n_topk,
                 m, consistent):
        super(GraphCon, self).__init__()
        self.encoder_q = base_encoder(arch, pretrained)
        self.agg_q = base_agg(attention, graph, n_topk, mode)
        if consistent:
            self.encoder_k = base_encoder(arch, pretrained)
            self.agg_k = base_agg(attention, graph, n_topk, mode)
            self.m = m

    def _momentum_update_key_backbone_and_agg(self):
        """
        Momentum update of the key backbone&agg
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.agg_q.parameters(), self.agg_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def forward(self, im_q, batch):
        fea_q, self_feat_q = self.encoder_q(im_q)
        Y_prob_q, Attention_q, Affinity_q, (Ga_q, Gg_q, G_q) = self.agg_q(fea_q, batch)
        return Y_prob_q, self_feat_q, Attention_q, Affinity_q, (Ga_q, Gg_q, G_q)

    def consistent_forward(self, im_k, batch):
        with torch.no_grad():
            self._momentum_update_key_backbone_and_agg()
            fea_k, self_feat_k = self.encoder_k(im_k)
            _, Attention_k, Affinity_k, (Ga_k, Gg_k, G_k) = self.agg_q(fea_k, batch)
        return None, self_feat_k, Attention_k, Affinity_k, (Ga_k, Gg_k, G_k)


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
