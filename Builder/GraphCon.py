import torch
import torch.nn as nn
class GraphCon(nn.Module):
    def __init__(self, mode, base_encoder, arch, pretrained,
                 base_agg, attention, graph, n_topk,
                 ema, momentum, consistent):
        super(GraphCon, self).__init__()
        self.encoder_q = base_encoder(arch, pretrained)
        self.agg_q = base_agg(attention, graph, n_topk, mode)
        self.momentum = momentum
        if consistent:
            self.encoder_k = base_encoder(arch, pretrained)
            self.agg_k = base_agg(attention, graph, n_topk, mode)
            self.ema = ema

    @torch.no_grad()
    def _momentum_update_key_backbone_and_agg(self):
        """
        Momentum update of the key backbone&agg
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.ema + param_q.data * (1. - self.ema)
        for param_q, param_k in zip(self.agg_q.parameters(), self.agg_k.parameters()):
            param_k.data = param_k.data * self.ema + param_q.data * (1. - self.ema)

    @torch.no_grad()
    def _momentum_update_key_backbone(self):

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
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
    def _batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle] 

    def forward(self, im_q, batch):
        fea_q, self_feat_q = self.encoder_q(im_q)
        self_feat_q = torch.nn.functional.normalize(self_feat_q, dim=1)
        Y_prob_q, Attention_q, Affinity_q, (Ga_q, Gg_q, G_q) = self.agg_q(fea_q, batch)
        return Y_prob_q, self_feat_q, Attention_q, Affinity_q, (Ga_q, Gg_q, G_q)

    def consistent_forward(self, im_k, batch):
        with torch.no_grad():
            self._momentum_update_key_backbone_and_agg()
            fea_k, self_feat_k = self.encoder_k(im_k)
            _, Attention_k, Affinity_k, (Ga_k, Gg_k, G_k) = self.agg_q(fea_k, batch)
        return None, self_feat_k, Attention_k, Affinity_k, (Ga_k, Gg_k, G_k)

    def self_supervised_forward(self, self_im_k):
        with torch.no_grad():
            self._momentum_update_key_backbone()
            # shuffle for making use of BN
            self_im_k_first, self_im_k_last, idx_unshuffle_first, idx_unshuffle_last = self._batch_shuffle(self_im_k)
            self_feat_k_first = self.encoder_k(self_im_k_first)
            self_feat_k_first = torch.nn.functional.normalize(self_feat_k_first, dim=1)
            self_feat_k_last = self.encoder_k(self_im_k_last)
            self_feat_k_last = torch.nn.functional.normalize(self_feat_k_last, dim=1)
            # undo shuffle
            self_feat_k_first = self._batch_unshuffle(self_feat_k_first, idx_unshuffle_first)
            self_feat_k_last = self._batch_unshuffle(self_feat_k_last, idx_unshuffle_last)
            self_feat_k = torch.cat((self_feat_k_first, self_feat_k_last), 0)
        return self_feat_k

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
