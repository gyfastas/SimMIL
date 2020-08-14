import faiss
import torch
import numpy as np
import math

DEFAULT_KMEANS_SEED = 1234

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)

def run_kmeans(x, nmb_clusters, verbose=False,
               seed=DEFAULT_KMEANS_SEED, gpu_device=0):
    """
    Runs kmeans on 1 GPU.

    Args:
    -----
    x: data
    nmb_clusters (int): number of clusters

    Returns:
    --------
    list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    # clus = faiss.Clustering(d, nmb_clusters)
    # clus.niter = 20
    # clus.max_points_per_centroid = 10000000
    # clus.seed = seed
    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.useFloat16 = False
    # flat_config.device = gpu_device

    # index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus = faiss.Kmeans(d, nmb_clusters, niter=20, verbose=False, gpu=True)
    clus.max_points_per_centroid = 10000000
    clus.seed = seed

    clus.train(x)

    # clus.train(x, index)
    _, I = clus.index.search(x, 1)
    losses = clus.obj
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

def run_kmeans_multi_gpu(x, nmb_clusters, verbose=False,
                         seed=DEFAULT_KMEANS_SEED, gpu_device=0):
    """
    Runs kmeans on multi GPUs.

    Args:
    -----
    x: data
    nmb_clusters (int): number of clusters

    Returns:
    --------
    list: ids of data in each cluster
    """
    n_data, d = x.shape
    ngpus = len(gpu_device)
    assert ngpus > 1

    # faiss implementation of k-means
    # clus = faiss.Clustering(d, nmb_clusters)
    # clus.niter = 20
    # clus.max_points_per_centroid = 10000000
    # clus.seed = seed
    # res = [faiss.StandardGpuResources() for i in range(ngpus)]
    # flat_config = []
    # for i in gpu_device:
    #     cfg = faiss.GpuIndexFlatConfig()
    #     cfg.useFloat16 = False
    #     cfg.device = i
    #     flat_config.append(cfg)

    # indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpus)]
    # index = faiss.IndexReplicas()
    # for sub_index in indexes:
    #     index.addIndex(sub_index)

    # perform the training

    clus = faiss.Kmeans(d, nmb_clusters, niter=20, verbose=False, gpu=True)
    clus.max_points_per_centroid = 10000000
    clus.seed = seed

    clus.train(x)
    # clus.train(x, index)
    _, I = clus.index.search(x, 1)
    losses = clus.obj
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

def _init_cluster_labels(config, data_len):
    no_kmeans_k = config.loss_params.n_kmeans  # how many wil be train

    # initialize cluster labels
    cluster_labels = torch.arange(data_len).long()
    cluster_labels = cluster_labels.unsqueeze(0).repeat(no_kmeans_k, 1)
    broadcast_cluster_labels = torch.cuda.comm.broadcast(cluster_labels, config.gpu_device)
    return broadcast_cluster_labels
