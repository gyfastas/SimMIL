import torch
import torch.nn as nn
class GraphCon(nn.Module):
    def __init__(self, mode, base_encoder, arch, pretrained,
                 base_agg, attention, graph, n_topk,
                 m, T,
                 dim, K, bag_label):
        super(GraphCon, self).__init__()
        self.encoder_q = base_encoder(arch, pretrained)
        self.agg_q = base_agg(attention, graph, n_topk, mode)
        self.encoder_k = base_encoder(arch, pretrained)
        self.agg_k = base_agg(attention, graph, n_topk, mode)

        self.m = m
        self.T = T
        self.bag_label = bag_label
        # create the queue
        self.register_buffer("bag_feature", torch.randn(dim, K))
        self.bag_feature = nn.functional.normalize(self.bag_feature, dim=0)

    def _momentum_update_key_backbone_and_agg(self):
        """
        Momentum update of the key backbone&agg
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.agg_q.parameters(), self.agg_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def forward(self, im_q, im_k, batch, bag_idx, label):
        fea_q, self_fea_q = self.encoder_q(im_q)
        Y_prob_q, Attention_q, Affinity_q, bag_feature_q = self.agg_q(fea_q, batch)
        q = nn.functional.normalize(bag_feature_q, dim=1)
        if im_k is None:
            return Y_prob_q
        with torch.no_grad():
            self._momentum_update_key_backbone_and_agg()
            fea_k, self_fea_k = self.encoder_k(im_k)
            _, Attention_k, Affinity_k, bag_feature_k = self.agg_q(fea_k, batch)
            k = nn.functional.normalize(bag_feature_k, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.bag_feature.clone().detach()])
        mask = torch.eq(*torch.meshgrid(label, self.bag_label)).long()
        l_neg = l_neg.masked_fill(mask == 1, -1e9)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._updata_bag_feature(q, bag_idx)
        # predictions, self_fea [N, 128], (bag logits, labels) for contrastive, attention weights, affinity weights
        # Here the bag logits are calculated on the attention weighted bag feature
        return Y_prob_q, (logits, labels), (self_fea_q, self_fea_k), (Attention_q, Attention_k), (Affinity_q, Affinity_k)

    def _updata_bag_feature(self, feature, bag_idx):
        self.bag_feature[:, bag_idx] = feature.T



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
