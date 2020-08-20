from functools import wraps
import copy
# from main_graphcon import bag_label

import torch
import torch.nn as nn

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

class MLP(torch.nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class NetWrapper(torch.nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size):
        super(NetWrapper, self).__init__()
        
        self.net = net
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def forward(self, x):
        representation, _ = self.net(x)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection


class GraphCon(nn.Module):
    def __init__(self, mode, base_encoder, arch, pretrained,
                 base_agg, attention, graph, n_topk,
                 m, T, dim, K, bag_label,
                 BYOL_flag=False, BYOL_size=None, BYOL_hidden_size=None):
        super(GraphCon, self).__init__()

        self.flag = BYOL_flag
        if self.flag == False:
            self.encoder_q = base_encoder(arch, pretrained)
            self.agg_q = base_agg(attention, graph, n_topk, mode)
            self.encoder_k = base_encoder(arch, pretrained)
            self.agg_k = base_agg(attention, graph, n_topk, mode)
        else:
            self.encoder_q = NetWrapper(base_encoder(arch, pretrained), BYOL_size, BYOL_hidden_size) # online
            self.encoder_k = None # target
            self.predictor_q = MLP(BYOL_size, BYOL_size, BYOL_hidden_size) # online
            
            self.agg_q = base_agg(attention, graph, n_topk, mode)
            self.agg_k = base_agg(attention, graph, n_topk, mode)
            
        self.ema = m
        self.T = T
        self.bag_label = bag_label
        # create the queue
        self.register_buffer("bag_feature", torch.randn(dim, K))
        self.bag_feature = nn.functional.normalize(self.bag_feature, dim=0)
        self.classifier = nn.Linear(128, 4)

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
            param_k.data = param_k.data * self.ema + param_q.data * (1. - self.ema)
    
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

    def _updata_bag_feature(self, feature, bag_idx):
        self.bag_feature[:, bag_idx] = feature.T

    @singleton('encoder_k')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.encoder_q)
        return target_encoder

    def _loss_fn(self, x, y):
        x = nn.functional.normalize(x, dim=-1, p=2)
        y = nn.functional.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, im_q, im_k, batch, bag_idx, label):

        if(self.flag==True):

            self_fea_q=None # do not need it
            self_fea_k=None

            fea_q = self.encoder_q(im_q) #one
            online_pred_one = self.predictor_q(fea_q)
            Y_prob_q, Attention_q, Affinity_q, bag_feature_q = self.agg_q(online_pred_one, batch)
            online_prob_one = self.classifier(online_pred_one)
            # Y_prob_q_sf = torch.softmax(Y_prob_q/0.07, dim=1)
            # Y_prob_q_sf =Y_prob_q
            q = nn.functional.normalize(bag_feature_q, dim=1)

            if im_k is None:
                return Y_prob_q
            
            fea_k = self.encoder_q(im_k) #two
            online_pred_two = self.predictor_q(fea_k)
            online_prob_two = self.classifier(online_pred_two)

            with torch.no_grad():
                target_encoder = self._get_target_encoder()
                # shuffle for making use of BN
                im_k_first, im_k_last, idx_unshuffle_first, idx_unshuffle_last = self._batch_shuffle(im_k)
                im_q_first, im_q_last, idx_unshuffle_first_q, idx_unshuffle_last_q = self._batch_shuffle(im_q)
                target_proj_one_first = target_encoder(im_k_first)
                target_proj_one_last  = target_encoder(im_k_last)
                target_proj_two_first = target_encoder(im_q_first)
                target_proj_two_last  = target_encoder(im_q_last)
                # undo shuffle
                target_proj_one_first, _ = self._batch_unshuffle(target_proj_one_first, target_proj_one_first, idx_unshuffle_first)
                target_proj_one_last , _ = self._batch_unshuffle(target_proj_one_last, target_proj_one_last, idx_unshuffle_last)
                target_proj_one = torch.cat((target_proj_one_first, target_proj_one_last), 0)
                target_proj_two_first, _ = self._batch_unshuffle(target_proj_two_first, target_proj_two_first, idx_unshuffle_first_q)
                target_proj_two_last , _ = self._batch_unshuffle(target_proj_two_last, target_proj_two_last, idx_unshuffle_last_q)
                target_proj_two = torch.cat((target_proj_two_first, target_proj_two_last), 0)
                # self_fea_k = torch.cat((self_feat_k_first, self_feat_k_last), 0)

            # with torch.no_grad():
            #     target_proj_one = target_encoder(im_q)
            #     target_proj_two = target_encoder(im_k)

            # ins_loss
            loss_one = self._loss_fn(online_pred_one, target_proj_one.detach())
            loss_two = self._loss_fn(online_pred_two, target_proj_two.detach())
            ins_loss = (loss_one + loss_two).mean()
            
            
            with torch.no_grad():
                self._momentum_update_key_backbone_and_agg()
                # shuffle for making use of BN
                # im_k_first, im_k_last, idx_unshuffle_first, idx_unshuffle_last = self._batch_shuffle(im_k)
                # feat_k_first, self_feat_k_first = self.encoder_k(im_k_first)
                # feat_k_last, self_feat_k_last = self.encoder_k(im_k_last)
                # undo shuffle
                # feat_k_first, self_feat_k_first = self._batch_unshuffle(feat_k_first, self_feat_k_first, idx_unshuffle_first)
                # feat_k_last , self_feat_k_last  = self._batch_unshuffle(feat_k_last, self_feat_k_last, idx_unshuffle_last)
                # fea_k = torch.cat((feat_k_first, feat_k_last), 0)
                # self_fea_k = torch.cat((self_feat_k_first, self_feat_k_last), 0)

                # fea_k, self_fea_k = self.encoder_k(im_k)
                Y_prob_k, Attention_k, Affinity_k, bag_feature_k = self.agg_k(target_proj_one, batch)
                # Y_prob_k_sf = torch.softmax(Y_prob_k/0.07, dim=1)
                # Y_prob_k_sf = Y_prob_k
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
            # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            labels=[]
            for i in label:
                labels.extend([i.item()] * 48)
            labels = torch.tensor(labels).long().cuda()

            self._updata_bag_feature(q, bag_idx)

            # print(torch.mean(Attention_q,dim=0))
            # print(torch.max(Attention_q, dim=0))

            # predictions, self_fea [N, 128], (bag logits, labels) for contrastive, attention weights, affinity weights
            # Here the bag logits are calculated on the attention weighted bag feature
            return Y_prob_q, (online_prob_one, online_prob_two), ins_loss, (logits, labels), (Attention_q, Attention_k), (Affinity_q, Affinity_k)
        
        else:
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
                _, Attention_k, Affinity_k, bag_feature_k = self.agg_k(fea_k, batch)
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
