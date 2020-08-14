"""
Local Aggregation Objective as defined in
https://arxiv.org/abs/1903.12355

Code is based on Tensorflow implementation:
https://github.com/neuroailab/LocalAggregation
"""
import torch
import numpy as np
import math

from torch import nn
import MemoryBank.mb_utils as utils

DEFAULT_KMEANS_SEED = 1234

class MemoryBank(object):
    """For efficiently computing the background vectors."""

    def __init__(self, size, dim, device_ids):
        self.size = size
        self.dim = dim
        self.device = torch.device("cuda:{}".format(device_ids[0]))
        self._bank = self._create()
        self.bank_broadcast = torch.cuda.comm.broadcast(self._bank, device_ids)
        self.device = [_bank.device for _bank in self.bank_broadcast]
        self.num_device = len(self.device)
        del self._bank
        # print(colored('Warning: using in-place scatter in memory bank update function', 'red'))

    def _create(self):
        # initialize random weights
        mb_init = torch.rand(self.size, self.dim, device=self.device)
        std_dev = 1. / np.sqrt(self.dim / 3)
        mb_init = mb_init * (2 * std_dev) - std_dev
        # L2 normalise so that the norm is 1
        mb_init = utils.l2_normalize(mb_init, dim=1)
        return mb_init.detach()  # detach so its not trainable

    def as_tensor(self):
        return self.bank_broadcast[0]

    def at_idxs(self, idxs):
        return torch.index_select(self.bank_broadcast[0], 0, idxs)

    def get_all_dot_products(self, vec):
        # [bs, dim]
        assert len(vec.size()) == 2
        return torch.matmul(vec, torch.transpose(self.bank_broadcast[0], 1, 0))

    def get_dot_products(self, vec, idxs):
        vec_shape = list(vec.size())  # [bs, dim]
        idxs_shape = list(idxs.size())  # [bs, ...]

        assert len(idxs_shape) in [1, 2]
        assert len(vec_shape) == 2
        assert vec_shape[0] == idxs_shape[0]

        if len(idxs_shape) == 1:
            with torch.no_grad():
                memory_vecs = torch.index_select(self._bank, 0, idxs)
                memory_vecs_shape = list(memory_vecs.size())
                assert memory_vecs_shape[0] == idxs_shape[0]
        else:  # len(idxs_shape) == 2
            with torch.no_grad():
                batch_size, k_dim = idxs.size(0), idxs.size(1)
                flat_idxs = idxs.view(-1)
                memory_vecs = torch.index_select(self._bank, 0, flat_idxs)
                memory_vecs = memory_vecs.view(batch_size, k_dim, self._bank.size(1))
                memory_vecs_shape = list(memory_vecs.size())

            vec_shape[1:1] = [1] * (len(idxs_shape) - 1)
            vec = vec.view(vec_shape)  # [bs, 1, dim]

        prods = memory_vecs * vec
        assert list(prods.size()) == memory_vecs_shape

        return torch.sum(prods, dim=-1)

    def update(self, indices, data_memory):
        # in lieu of scatter-update operation
        data_dim = data_memory.size(1)
        data_memory = data_memory.detach()
        indices = indices.unsqueeze(1).repeat(1, data_dim)

        for i in range(self.num_device):
            if i > 0:
                # start.record()
                device = self.device[i]
                indices = indices.to(device)
                data_memory = data_memory.to(device)
            self.bank_broadcast[i] = self.bank_broadcast[i].scatter_(0, indices, data_memory)

    def synchronization_check(self):
        for i in range(len(self.bank_broadcast)):
            if i == 0:
                device = self.bank_broadcast[0].device
            else:
                assert torch.equal(self.bank_broadcast[0], self.bank_broadcast[i].to(device))

class Kmeans(object):
    """
    Train <k> different k-means clusterings with different
    random seeds. These will be used to compute close neighbors
    for a given encoding.
    """

    def __init__(self, k, memory_bank, gpu_device=0):
        super().__init__()
        self.k = k
        self.memory_bank = memory_bank
        self.gpu_device = gpu_device

    def compute_clusters(self):
        """
        Performs many k-means clustering.

        Args:
            x_data (np.array N * dim): data to cluster
        """
        data = self.memory_bank.as_tensor()
        data_npy = data.cpu().detach().numpy()
        clusters = self._compute_clusters(data_npy)
        return clusters

    def _compute_clusters(self, data):
        pred_labels = []
        for k_idx, each_k in enumerate(self.k):
            # cluster the data

            if len(self.gpu_device) == 1:  # single gpu
                I, _ = utils.run_kmeans(data, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
                                  gpu_device=self.gpu_device[0])
            else:  # multigpu
                I, _ = utils.run_kmeans_multi_gpu(data, each_k, seed=k_idx + DEFAULT_KMEANS_SEED,
                                            gpu_device=self.gpu_device)

            clust_labels = np.asarray(I)
            pred_labels.append(clust_labels)
        pred_labels = np.stack(pred_labels, axis=0)
        pred_labels = torch.from_numpy(pred_labels).long()

        return pred_labels

class InstanceDiscriminationLossModule(torch.nn.Module):
    def __init__(self, memory_bank_broadcast, cluster_labels_broadcast=None, k=4096, t=0.07, m=0.5):
        super(InstanceDiscriminationLossModule, self).__init__()
        self.k, self.t, self.m = k, t, m

        self.indices = None
        self.outputs = None
        self._bank = None  # pass in via forward function
        self.memory_bank_broadcast = memory_bank_broadcast
        self.data_len = memory_bank_broadcast[0].size(0)

    def _softmax(self, dot_prods):
        Z = 2876934.2 / 1281167 * self.data_len
        return torch.exp(dot_prods / self.t) / Z

    def updated_new_data_memory(self, indices, outputs):
        outputs = utils.l2_normalize(outputs)
        data_memory = torch.index_select(self._bank, 0, indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * outputs
        return utils.l2_normalize(new_data_memory, dim=1)

    def synchronization_check(self):
        for i in range(len(self.memory_bank_broadcast)):
            if i == 0:
                device = self.memory_bank_broadcast[0].device
            else:
                assert torch.equal(self.memory_bank_broadcast[0], self.memory_bank_broadcast[i].to(device))

    def _get_dot_products(self, vec, idxs):
        """
        This function is copied from the get_dot_products in Memory_Bank class
        Since we want to register self._bank as a buffer (to be broadcasted to multigpus) instead of self.memory_bank,
        we need to avoid calling self.memory_bank get_dot_products

        """
        vec_shape = list(vec.size())  # [bs, dim]
        idxs_shape = list(idxs.size())  # [bs, ...]

        assert len(idxs_shape) in [1, 2]
        assert len(vec_shape) == 2
        assert vec_shape[0] == idxs_shape[0]

        if len(idxs_shape) == 1:
            with torch.no_grad():
                memory_vecs = torch.index_select(self._bank, 0, idxs)
                memory_vecs_shape = list(memory_vecs.size())
                assert memory_vecs_shape[0] == idxs_shape[0]
        else:  # len(idxs_shape) == 2
            with torch.no_grad():
                batch_size, k_dim = idxs.size(0), idxs.size(1)
                flat_idxs = idxs.view(-1)
                memory_vecs = torch.index_select(self._bank, 0, flat_idxs)
                memory_vecs = memory_vecs.view(batch_size, k_dim, self._bank.size(1))
                memory_vecs_shape = list(memory_vecs.size())

            vec_shape[1:1] = [1] * (len(idxs_shape) - 1)
            vec = vec.view(vec_shape)  # [bs, 1, dim]

        prods = memory_vecs * vec
        assert list(prods.size()) == memory_vecs_shape

        return torch.sum(prods, dim=-1)

    def compute_data_prob(self):
        logits = self._get_dot_products(self.outputs, self.indices)
        return self._softmax(logits)

    def compute_noise_prob(self):
        batch_size = self.indices.size(0)
        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k),
                                   device=self.outputs.device)  # U(0, data_len)
        noise_indx = noise_indx.long()
        logits = self._get_dot_products(self.outputs, noise_indx)
        noise_probs = self._softmax(logits)
        return noise_probs

    def forward(self, indices, outputs, gpu_idx):
        self.indices = indices.detach()
        self.outputs = utils.l2_normalize(outputs, dim=1)
        self._bank = self.memory_bank_broadcast[gpu_idx]

        batch_size = self.indices.size(0)
        data_prob = self.compute_data_prob()
        noise_prob = self.compute_noise_prob()

        assert data_prob.size(0) == batch_size
        assert noise_prob.size(0) == batch_size
        assert noise_prob.size(1) == self.k

        base_prob = 1.0 / self.data_len
        eps = 1e-7

        ## Pmt
        data_div = data_prob + (self.k * base_prob + eps)

        ln_data = torch.log(data_prob) - torch.log(data_div)

        ## Pon
        noise_div = noise_prob + (self.k * base_prob + eps)
        ln_noise = math.log(self.k * base_prob) - torch.log(noise_div)

        curr_loss = -(torch.sum(ln_data) + torch.sum(ln_noise))
        curr_loss = curr_loss / batch_size

        new_data_memory = self.updated_new_data_memory(self.indices, self.outputs)

        return curr_loss.unsqueeze(0), new_data_memory

class LocalAggregationLossModule(torch.nn.Module):

    def __init__(self, memory_bank_broadcast, cluster_label_broadcast, k=4096, t=0.07, m=0.5):
        super(LocalAggregationLossModule, self).__init__()
        self.k, self.t, self.m = k, t, m

        self.indices = None
        self.outputs = None
        self._bank = None  # pass in via forward function
        self._cluster_labels = None
        self.memory_bank_broadcast = memory_bank_broadcast
        self.cluster_label_broadcast = cluster_label_broadcast
        self.data_len = memory_bank_broadcast[0].size(0)
        self.first_iteration_kmeans = True

    def _softmax(self, dot_prods):
        Z = 2876934.2 / 1281167 * self.data_len
        return torch.exp(dot_prods / self.t) / Z

    def updated_new_data_memory(self, indices, outputs):
        outputs = utils.l2_normalize(outputs)
        data_memory = torch.index_select(self._bank, 0, indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * outputs
        return utils.l2_normalize(new_data_memory, dim=1)

    def synchronization_check(self):
        for i in range(len(self.memory_bank_broadcast)):
            if i == 0:
                device = self.memory_bank_broadcast[0].device
            else:
                assert torch.equal(self.memory_bank_broadcast[0], self.memory_bank_broadcast[i].to(device))

    def _get_all_dot_products(self, vec):
        assert len(vec.size()) == 2
        return torch.matmul(vec, torch.transpose(self._bank, 1, 0))

    def __get_close_nei_in_back(self, each_k_idx, cluster_labels,
                                back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][self.indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = utils.repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei.byte().cuda()

    def __get_relative_prob(self, all_close_nei, back_nei_probs):
        relative_probs = torch.sum(
            torch.where(
                all_close_nei,
                back_nei_probs,
                torch.zeros_like(back_nei_probs),
            ), dim=1)
        # normalize probs
        relative_probs = relative_probs / torch.sum(back_nei_probs, dim=1, keepdim=True)
        return relative_probs

    def __get_close_nei(self, each_k_idx, cluster_labels, indices):
        batch_size = self.indices.size(0)
        dtype = torch.int32  # convert to 32-bit integer to save memory consumption
        batch_labels = cluster_labels[each_k_idx][indices].to(dtype)
        _cluster_labels = cluster_labels[each_k_idx].to(dtype).unsqueeze(0).expand(batch_size, -1)
        batch_labels = utils.repeat_1d_tensor(batch_labels, _cluster_labels.size(1))
        curr_close_nei = torch.eq(batch_labels, _cluster_labels)
        return curr_close_nei.byte()

    def forward(self, indices, outputs, gpu_idx):
        """
        :param back_nei_idxs: shape (batch_size, 4096)
        :param all_close_nei: shape (batch_size, _size_of_dataset) in byte
        """
        self.indices = indices.detach()
        self.outputs = utils.l2_normalize(outputs, dim=1)
        self._bank = self.memory_bank_broadcast[gpu_idx]  # select a mem bank based on gpu device
        self._cluster_labels = self.cluster_label_broadcast[gpu_idx]

        k = self.k

        all_dps = self._get_all_dot_products(self.outputs)
        back_nei_dps, back_nei_idxs = torch.topk(all_dps, k=k, sorted=False, dim=1)
        back_nei_probs = self._softmax(back_nei_dps)

        all_close_nei_in_back = None
        no_kmeans = self._cluster_labels.size(0)
        with torch.no_grad():
            for each_k_idx in range(no_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(
                    each_k_idx, self._cluster_labels, back_nei_idxs, k)

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    # assuming all_close_nei and curr_close_nei are byte tensors
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        relative_probs = self.__get_relative_prob(all_close_nei_in_back, back_nei_probs)
        loss = -torch.mean(torch.log(relative_probs + 1e-7)).unsqueeze(0)

        # compute new data memory
        new_data_memory = self.updated_new_data_memory(self.indices, self.outputs)

        return loss, new_data_memory

class MocoLossModule(torch.nn.Module):
    def __init__(self, memory_bank_broadcast, k=4096, t=0.07):
        super(MocoLossModule, self).__init__()
        
        self.k = k
        self.t = t
        self.indices = None
        self.fea_q = None
        self.fea_k = None
        self._bank = None  # pass in via forward function
        self.memory_bank_broadcast = memory_bank_broadcast
        self.data_len = memory_bank_broadcast[0].size(0)

    # def _softmax(self, dot_prods):

    #     return torch.exp(dot_prods / self.t)

    # def updated_new_data_memory(self, indices, outputs):

    #     data_memory = torch.index_select(self._bank, 0, indices)
    #     new_data_memory = data_memory * self.m + (1 - self.m) * outputs
    #     return utils.l2_normalize(new_data_memory, dim=1)

    def _get_dot_products(self, vec, idxs):
        """
        This function is copied from the get_dot_products in Memory_Bank class
        Since we want to register self._bank as a buffer (to be broadcasted to multigpus) instead of self.memory_bank,
        we need to avoid calling self.memory_bank get_dot_products

        """
        vec_shape = list(vec.size())  # [bs, dim]
        idxs_shape = list(idxs.size())  # [bs, ...]

        assert len(idxs_shape) in [1, 2]
        assert len(vec_shape) == 2
        assert vec_shape[0] == idxs_shape[0]

        if len(idxs_shape) == 1:
            with torch.no_grad():
                memory_vecs = torch.index_select(self._bank, 0, idxs)
                memory_vecs_shape = list(memory_vecs.size())
                assert memory_vecs_shape[0] == idxs_shape[0]
        else:  # len(idxs_shape) == 2
            with torch.no_grad():
                batch_size, k_dim = idxs.size(0), idxs.size(1)
                flat_idxs = idxs.view(-1)
                memory_vecs = torch.index_select(self._bank, 0, flat_idxs)
                memory_vecs = memory_vecs.view(batch_size, k_dim, self._bank.size(1))
                memory_vecs_shape = list(memory_vecs.size())

            vec_shape[1:1] = [1] * (len(idxs_shape) - 1)
            vec = vec.view(vec_shape)  # [bs, 1, dim]

        prods = memory_vecs * vec
        assert list(prods.size()) == memory_vecs_shape

        return torch.sum(prods, dim=-1)
    
    def compute_data_prob(self):
        logits = torch.einsum('nc,nc->n', [self.fea_q, self.fea_k]).unsqueeze(-1)
        # data_probs = self._softmax(logits)
        return logits

    def compute_noise_prob(self):
        batch_size = self.indices.size(0)
        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k),
                                   device=self.fea_q.device)  # U(0, data_len)
        noise_indx = noise_indx.long()
        logits = self._get_dot_products(self.fea_q, noise_indx)
        # noise_probs = self._softmax(logits)
        return logits

    def forward(self,idx, fea_q, fea_k, gpu_idx):
        
        self.indices = idx.detach()
        self.fea_q = torch.nn.functional.normalize(fea_q, dim=1)
        self.fea_k = torch.nn.functional.normalize(fea_k, dim=1)
        self._bank = self.memory_bank_broadcast[gpu_idx]
        
        batch_size = self.indices.size(0)
        # positive logits: Nx1
        data_logits = self.compute_data_prob()
        # negative logits: NxK
        noise_logits = self.compute_noise_prob()
        # logits: Nx(1+K)
        logits = torch.cat([data_logits, noise_logits], dim=1)
        assert data_logits.size(0) == batch_size
        assert noise_logits.size(0) == batch_size
        assert noise_logits.size(1) == self.k
        # apply temperature
        logits /= self.t
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        criterion = nn.CrossEntropyLoss()
        loss =criterion(logits, labels)

        # loss = - torch.sum(torch.log(data_prob)) + torch.sum(torch.log(noise_prob))
        # loss = loss/batch_size

        # new_data_memory = self.updated_new_data_memory(self.indices, self.fea_k)

        return loss, self.fea_k