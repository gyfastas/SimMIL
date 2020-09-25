import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from Builder.e2e import SelfAttention
from GraphMIL.GraphBuilder.KNNGraphBuilder import KNNGraphBuilder
from GraphMIL.GCN.GCNConv import GCNConv
from GraphMIL.Aggregator.GraphWeightedAvgPooling import GraphWeightedAvgPooling
from torch_geometric.utils import dense_to_sparse
from Builder.cifar_res import ResNet32x32, ShakeShakeBlock, cifar_shakeshake26

from Builder.memorybank import ODCMemory
import numpy as np
from Builder.WRN import Wide_ResNet

from torch.nn.functional import one_hot

# Attention/Mean Pooling
def attention_weighted_feature(H, A, batch, pool_model='mean'):

    if batch is None:
        A = F.softmax(A, dim=1)  # softmax over N
        attention_weighted_feature = torch.mm(A, H) 
    elif batch.shape[0] == H.shape[0]:
        bag_num = torch.max(batch) + 1
        indicator = F.one_hot(batch, bag_num).float()
        if pool_model == 'mean':
            A = torch.softmax(indicator.sub((1 - indicator).mul(65535)), dim=0)
        elif pool_model == 'attention':
            A = torch.mul(torch.transpose(A, 1, 0), indicator)
            A = torch.softmax(A.sub((1 - indicator).mul(65535)), dim=0)

        attention_weighted_feature = torch.mm(A.T, H)
    return attention_weighted_feature

def bag_shffule(indicator, bag_label):
    # shuffle bag number
    index = torch.randperm(indicator.shape[1]).cuda()
    shuffle_indicator = indicator[:, index]
    shuffle_label = bag_label[index]
    # random mask for dropout some instances in negative bags
    random_mask1 = torch.randint(0, 2, indicator.shape).cuda()
    random_mask2 = torch.randint(0, 2, indicator.shape).cuda()
    random_mask1[:, bag_label==1] = 1
    random_mask2[:, shuffle_label==1] = 1
    indicator = indicator*random_mask1
    shuffle_indicator = shuffle_indicator*random_mask2
    # mix the attention mask and labels
    mixed_indicator = indicator + shuffle_indicator
    mixed_label = bag_label + shuffle_label
    mixed_indicator[mixed_indicator > 1] = 1
    mixed_label[mixed_label > 1] = 1
    return mixed_label, mixed_indicator

class Res_Cls_Moco(nn.Module):
    '''
    Cifar classification + MoCod
    '''
    def __init__(self, num_classes, m, attention, L= 384, dim=128, K=4096, T=0.07):
        super(Res_Cls_Moco, self).__init__()
        # Student Net
        self.STU = cifar_shakeshake26(attention=attention, L=L)
        # self.STU.attention = attention(L)
        self.STU.fc1 = nn.Linear(384, num_classes)
        self.STU.fc2 = nn.Linear(384, dim)
        # Teacher Net
        self.TEA = cifar_shakeshake26(attention=attention, L=L)
        # self.TEA.attention = attention(L)
        self.TEA.fc1 = nn.Linear(384, num_classes)
        self.TEA.fc2 = nn.Linear(384, dim)
        # Memory Queue
        self.K = K
        self.T = T
        self.m = m
        # self.register_buffer('ins_labels_mmb', F.one_hot(init_ins_labels, 2).float())
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _momentum_update_tea_and_stu(self):
        """
        Momentum update of the key backbone&agg
        """
        for param_q, param_k in zip(self.STU.parameters(), self.TEA.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) < self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = ptr + batch_size
        else:
            self.queue[:, ptr:] = keys[:self.K-ptr].T
            self.queue[:, :ptr + batch_size - self.K] = keys[self.K-ptr:].T
            ptr = ptr + batch_size - self.K
        # ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        # print(ptr)

    @staticmethod
    def get_shuffle_ids(bsz):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def forward(self, im_q, im_k):
        # bsz = im_q.size(0)
        ins_feat_q, _ = self.STU(im_q, None)
        bag_cls_q = self.STU.fc1(ins_feat_q)
        ins_embed_q = self.STU.fc2(ins_feat_q)
        q = nn.functional.normalize(ins_embed_q, dim=1)
        if im_k is None:
            return (bag_cls_q, None), (None, None)
        with torch.no_grad():
            self._momentum_update_tea_and_stu()
            #TODO: shuffle may not be suitable for bag

            # ids for ShuffleBN
            # shuffle_ids, reverse_ids = self.get_shuffle_ids(bsz)
            # #shuffle
            # im_k = im_k[shuffle_ids]
            # ins_feat_k, attention_k = self.TEA(im_k, batch)
            # #unshuffle
            # ins_feat_k, attention_k = ins_feat_k[reverse_ids], attention_k[reverse_ids]

            # forward twice

            ins_feat_k_0, _ = self.TEA(im_k[:int(im_k.size(0)/2)], None)
            ins_feat_k_1, _ = self.TEA(im_k[int(im_k.size(0)/2):], None)
            ins_feat_k = torch.cat((ins_feat_k_0, ins_feat_k_1))
            bag_cls_k = self.TEA.fc1(ins_feat_k)
            ins_embed_k = self.TEA.fc2(ins_feat_k)
            k = nn.functional.normalize(ins_embed_k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k)
        return (bag_cls_q, bag_cls_k), (logits, labels)

class MT_ResAMIL(nn.Module):
    '''
    MIL + MoCo with Instance or Bag contrast
    '''
    def __init__(self, num_classes, attention, L, m, init_ins_labels, dim=128, K=4096, T=0.07):
        super(MT_ResAMIL, self).__init__()
        # Student Net
        self.STU = cifar_shakeshake26(attention=attention, L=L)
        self.STU.fc1 = nn.Linear(384, num_classes)
        self.STU.fc2 = nn.Linear(384, dim)
        # Teacher Net
        self.TEA = cifar_shakeshake26(attention=attention, L=L)
        self.TEA.fc1 = nn.Linear(384, num_classes)
        self.TEA.fc2 = nn.Linear(384, dim)
        # Memory Queue
        self.K = K
        self.T = T
        self.m = m
        # self.register_buffer('ins_labels_mmb', F.one_hot(init_ins_labels, 2).float())
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _momentum_update_tea_and_stu(self):
        """
        Momentum update of the key backbone&agg
        """
        for param_q, param_k in zip(self.STU.parameters(), self.TEA.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) < self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = ptr + batch_size
        else:
            self.queue[:, ptr:] = keys[:self.K-ptr].T
            self.queue[:, :ptr + batch_size - self.K] = keys[self.K-ptr:].T
            ptr = ptr + batch_size - self.K
        # ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        print(ptr)

    @staticmethod
    def get_shuffle_ids(bsz):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    @staticmethod
    def bag_shuffle(batch):
        bag_num = torch.max(batch)
        shuffled_batch = torch.randint(0, bag_num+1, batch.shape).cuda()
        return shuffled_batch

    def forward(self, im_q, im_k, batch):
        # bsz = im_q.size(0)
        ins_feat_q, attention_q = self.STU(im_q, batch)
        bag_feat_q = attention_weighted_feature(ins_feat_q, attention_q, batch)
        bag_cls_q = self.STU.fc1(bag_feat_q)
        ins_embed_q = self.STU.fc2(ins_feat_q)
        q = nn.functional.normalize(ins_embed_q, dim=1)
        # shuffled_batch_list =[]
        # for _ in range(16):
        #     shuffled_batch = self.bag_shuffle(batch)
        #     bag_feat_tmp = self.attention_weighted_feature(ins_feat_q, attention_q, shuffled_batch)
        #     bag_feat_q = torch.cat((bag_feat_q, bag_feat_tmp))
        #     shuffled_batch_list.append(shuffled_batch)
        # bag_embed_q = self.STU.fc2(bag_feat_q)
        # q = nn.functional.normalize(bag_embed_q, dim=1)
        if im_k is None:
            return bag_cls_q
        with torch.no_grad():
            self._momentum_update_tea_and_stu()
            #TODO: shuffle may not be suitable for bag

            # ids for ShuffleBN
            # shuffle_ids, reverse_ids = self.get_shuffle_ids(bsz)
            # #shuffle
            # im_k = im_k[shuffle_ids]
            # ins_feat_k, attention_k = self.TEA(im_k, batch)
            # #unshuffle
            # ins_feat_k, attention_k = ins_feat_k[reverse_ids], attention_k[reverse_ids]

            # forward twice
            im_k_0, batch_0 = im_k[batch < batch.float().mean()], batch[batch < batch.float().mean()]
            ins_feat_k_0, attention_k_0 = self.TEA(im_k_0, batch_0)
            im_k_1, batch_1 = im_k[batch >= batch.float().mean()], batch[batch >= batch.float().mean()]
            ins_feat_k_1, attention_k_1 = self.TEA(im_k_1, batch_1)
            ins_feat_k = torch.cat((ins_feat_k_0, ins_feat_k_1))
            attention_k = torch.cat((attention_k_0, attention_k_1), dim=1)
            bag_feat_k = attention_weighted_feature(ins_feat_k, attention_k, batch)
            bag_cls_k = self.TEA.fc1(bag_feat_k)
            ins_embed_k = self.TEA.fc2(ins_feat_k)
            k = nn.functional.normalize(ins_embed_k, dim=1)
            # for shuffled_batch in shuffled_batch_list:
            #     bag_feat_tmp = self.attention_weighted_feature(ins_feat_k, attention_k, shuffled_batch)
            #     bag_feat_k = torch.cat((bag_feat_k, bag_feat_tmp))
            # bag_embed_k = self.TEA.fc2(bag_feat_k)
            # k = nn.functional.normalize(bag_embed_k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k)
        return (bag_cls_q, bag_cls_k), (logits, labels)

class MT_ResAMIL_TE(nn.Module):
    '''f
    MIL with temporal ensemble
    '''
    def __init__(self, num_classes, attention, L, m, init_ins_labels, ins_label_num):
        super(MT_ResAMIL_TE, self).__init__()
        # self.STU = cifar_shakeshake26(num_classes=num_classes, attention=attention, L=L)
        self.STU = Wide_ResNet(28, 2, 0.3, num_classes, attention, L)
        self.STU.fc1 = nn.Linear(L, num_classes)
        self.STU.fc2 = nn.Linear(L, ins_label_num)
        self.m = m
        self.register_buffer('ins_labels_mmb', F.one_hot(init_ins_labels, ins_label_num).float())
        self.register_buffer('ins_labels_mmb_revised', F.one_hot(init_ins_labels, ins_label_num).float())
        # self.register_buffer('ins_labels_mmb', torch.zeros(init_ins_labels.shape[0], ins_label_num).float())
        # self.register_buffer('ins_labels_mmb_revised', torch.zeros(init_ins_labels.shape[0], ins_label_num).float())
    def _updata_ins_mb(self, output, index, epoch):
        self.ins_labels_mmb[index] = self.ins_labels_mmb[index] * self.m + output * (1. - self.m)
        self.ins_labels_mmb_revised[index] = self.ins_labels_mmb[index]
        # self.ins_labels_mmb_revised[index] = self.ins_labels_mmb[index]/(1 - self.m**epoch)
        # print(self.ins_labels_mmb_revised[index])


    def forward(self, im_q, batch):
        # bsz = im_q.size(0)
        ins_feat_q, attention_q = self.STU(im_q, batch)
        bag_feat_q = attention_weighted_feature(ins_feat_q, attention_q, batch, 'mean')
        bag_cls = self.STU.fc1(bag_feat_q)
        ins_cls = self.STU.fc2(ins_feat_q)
        return bag_cls, ins_cls

class MT_ResAMIL_BagIns(nn.Module):
    '''
    MIL Baseline with both
        embedding-based (fc1)
        Instance-based (fc2)
    '''
    def __init__(self, num_classes, attention, L):
        super(MT_ResAMIL_BagIns, self).__init__()
        # Student Net
        self.STU = Wide_ResNet(28, 2, 0.3, num_classes, attention, L)
        self.STU.fc1 = nn.Linear(L, num_classes) #embedding cls
        self.STU.fc2 = nn.Linear(L, num_classes) #instance cls


    def forward(self, im_q, batch, pool_model, paradigm):
        # bsz = im_q.size(0)
        ins_feat_q, attention_q = self.STU(im_q, batch)
        if paradigm == 'embedding':
            bag_feat_q = attention_weighted_feature(ins_feat_q, attention_q, batch, pool_model)
            embedding_cls = self.STU.fc1(bag_feat_q)
            return embedding_cls
        elif paradigm == 'instance':
            ins_cls = self.STU.fc2(ins_feat_q)
            instance_cls = attention_weighted_feature(ins_cls, attention_q, batch, pool_model)
            return instance_cls

class MT_ResAMIL_ODC(nn.Module):
    '''
    MIL with temporal ensemble
    '''
    def __init__(self, num_classes, attention, L, m, dim, length, K):
        super(MT_ResAMIL_ODC, self).__init__()
        # self.STU = cifar_shakeshake26(num_classes=num_classes, attention=attention, L=L)
        self.STU = Wide_ResNet(28, 2, 0.3, num_classes, attention, L)
        self.STU.fc1 = nn.Linear(L, num_classes)
        self.STU.head = nn.Sequential(nn.Linear(L, dim), nn.ReLU())
        self.STU.fc2 = nn.Linear(dim, K)
        self.memory_bank = ODCMemory(length=length, feat_dim=dim, momentum=m,
                                     num_classes=K, min_cluster=20)
        self.K = K


    def forward(self, im_q, batch):
        # bsz = im_q.size(0)
        ins_feat, attention = self.STU(im_q, batch)
        bag_feat = attention_weighted_feature(ins_feat, attention, batch)
        bag_cls = self.STU.fc1(bag_feat)


        ins_feat_RD = self.STU.head(ins_feat)
        ins_cls = self.STU.fc2(ins_feat_RD)
        return bag_cls, ins_cls, ins_feat_RD

class ClusterAttentionPooling(nn.Module):
    """
    Cluster based attention pooling
    """
    def __init__(self, attpooling, num_clustering):
        self.attpooling = attpooling
        self.num_clustering = num_clustering
    
    def avg_clustering(self, ins_feat, cluster_labels, batch):
        if batch is None:
            ins_feat = ins_feat.unsqueeze(0).permute(1,0,2).repeat([1, cluster_labels.shape[1], 1]) #[N, M, F]
            return (ins_feat * cluster_labels.unsqueeze(-1)).mean(0) #[M, F]
        ##TODO: multi-bag support
        else:
            raise NotImplementedError
        

    def forward(self, ins_feat, cluster_labels, batch):
        """
        Args:
            ins_feat: [N, F] where N is the size of bag (or multi-bag)
            cluster_labels: [N,] or [N, M]
            batch: [N,]
        """
        ## Check if one-hot
        if cluster_labels.dim() == 2:
            pass
        elif cluster_labels.dim() == 1:
            ##make one-hot labels
            cluster_labels = one_hot(cluster_labels, self.num_clustering)
        else:
            raise Exception("Cluster labels should be of size [N,] or [N, M]")

        ins_feat = self.avg_clustering(ins_feat, cluster_labels, batch) #[N, F] => [M, F]
        ins_feat = self.attpooling(ins_feat) #[M, F] => [1, F]

        return ins_feat

    def extract(self, im_q, batch):
        with torch.no_grad():
            ins_feat, _ = self.STU(im_q, batch)
            ins_feat_RD = self.STU.head(ins_feat)
            return ins_feat_RD.cpu()

    def set_reweight(self, labels=None, reweight_pow=0.5):
        """Loss re-weighting.
        Re-weighting the loss according to the number of samples in each class.
        Args:
            labels (numpy.ndarray): Label assignments. Default: None.
            reweight_pow (float): The power of re-weighting. Default: 0.5.
        """
        if labels is None:
            if self.memory_bank.label_bank.is_cuda:
                labels = self.memory_bank.label_bank.cpu().numpy()
            else:
                labels = self.memory_bank.label_bank.numpy()
        hist = np.bincount(
            labels, minlength=self.K).astype(np.float32)
        inv_hist = (1. / (hist + 1e-5)) ** reweight_pow
        weight = inv_hist / inv_hist.sum()
        return torch.from_numpy(weight).cuda()
