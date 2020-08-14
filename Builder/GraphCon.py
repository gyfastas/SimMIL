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

        self.ema = m
        self.T = T
        self.bag_label = bag_label
        # create the queue
        self.register_buffer("bag_feature", torch.randn(dim, K))
        self.bag_feature = nn.functional.normalize(self.bag_feature, dim=0)

    @torch.no_grad()
    def _momentum_update_key_backbone_and_agg(self):
        """
        Momentum update of the key backbone&agg
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.ema + param_q.data * (1. - self.ema)
        for param_q, param_k in zip(self.agg_q.parameters(), self.agg_k.parameters()):
            param_k.data = param_k.data * self.ema + param_q.data * (1. - self.ema)

    # @torch.no_grad()
    # def _momentum_update_key_backbone(self):

    #     for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    #         param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        batch_size = x.shape[0]
        batch_first = int(batch_size/2)
        # random shuffle index
        idx_shuffle_first = torch.randperm(batch_first).cuda()
        idx_shuffle_last = torch.arange(batch_first,batch_size,step=1)
        idx_tmp = torch.randperm(batch_first)
        idx_shuffle_last = idx_shuffle_last[idx_tmp]
        # index for restoring
        idx_unshuffle_first = torch.argsort(idx_shuffle_first)
        idx_unshuffle_last = torch.argsort(idx_shuffle_last)
        # shuffled index
        return x[idx_shuffle_first], x[idx_shuffle_last], idx_unshuffle_first, idx_unshuffle_last


    @torch.no_grad()
    def _batch_unshuffle(self, x1, x2, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x1[idx_unshuffle], x2[idx_unshuffle]

    def forward(self, im_q, im_k, batch, bag_idx, label):
        fea_q, self_fea_q = self.encoder_q(im_q)
        Y_prob_q, Attention_q, Affinity_q, bag_feature_q = self.agg_q(fea_q, batch)
        q = nn.functional.normalize(bag_feature_q, dim=1)
        if im_k is None:
            return Y_prob_q
        with torch.no_grad():
            self._momentum_update_key_backbone_and_agg()
            # shuffle for making use of BN
            im_k_first, im_k_last, idx_unshuffle_first, idx_unshuffle_last = self._batch_shuffle(im_k)
            feat_k_first, self_feat_k_first = self.encoder_k(im_k_first)
            feat_k_last, self_feat_k_last = self.encoder_k(im_k_last)
            # undo shuffle
            feat_k_first, self_feat_k_first = self._batch_unshuffle(feat_k_first, self_feat_k_first, idx_unshuffle_first)
            feat_k_last , self_feat_k_last  = self._batch_unshuffle(feat_k_last, self_feat_k_last, idx_unshuffle_last)
            fea_k = torch.cat((feat_k_first, feat_k_last), 0)
            self_fea_k = torch.cat((self_feat_k_first, self_feat_k_last), 0)
            # fea_k, self_fea_k = self.encoder_k(im_k)
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


    # def self_supervised_forward(self, self_im_k):
    #     with torch.no_grad():
    #         self._momentum_update_key_backbone()
    #         # shuffle for making use of BN
    #         self_im_k_first, self_im_k_last, idx_unshuffle_first, idx_unshuffle_last = self._batch_shuffle(self_im_k)
    #         self_feat_k_first = self.encoder_k(self_im_k_first)
    #         self_feat_k_first = torch.nn.functional.normalize(self_feat_k_first, dim=1)
    #         self_feat_k_last = self.encoder_k(self_im_k_last)
    #         self_feat_k_last = torch.nn.functional.normalize(self_feat_k_last, dim=1)
    #         # undo shuffle
    #         self_feat_k_first = self._batch_unshuffle(self_feat_k_first, idx_unshuffle_first)
    #         self_feat_k_last = self._batch_unshuffle(self_feat_k_last, idx_unshuffle_last)
    #         self_feat_k = torch.cat((self_feat_k_first, self_feat_k_last), 0)
    #     return self_feat_k

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
