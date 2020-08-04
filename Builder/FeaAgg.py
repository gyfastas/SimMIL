import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .e2e import SelfAttention
from GraphMIL.GraphBuilder.KNNGraphBuilder import KNNGraphBuilder
from GraphMIL.GCN.GCNConv import GCNConv
from GraphMIL.Aggregator.GraphWeightedAvgPooling import GraphWeightedAvgPooling
from torch_geometric.utils import dense_to_sparse



#2. backbone+Agg
# 2.1 Backbone
class ResBackbone(nn.Module):
    def __init__(self, arch, pretrained):
        super(ResBackbone, self).__init__()
        backbone = models.__dict__[arch](pretrained=pretrained)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, 128)
    def forward(self, x):
        # x = x.squeeze(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        self_feat = self.fc(x)
        return x, self_feat
# 2.2 Agg
class AggNet(nn.Module):
    def __init__(self, attention, graph, n_topk):
        super(AggNet, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.attention = attention()
        if graph is not None:
            self.graph = graph(n_topk)
        else:
            self.graph = graph
        self.classifier = nn.Linear(self.L*self.K, 4)

    def forward(self, H, batch):
        attention = self.attention(H)
        preds, Affinity, G = self.multi_batch(H, attention, batch)
        return preds, Affinity, G

    def multi_batch(self, H, A, batch):
        ## batch: map each instance to corresponding bag in a mini-batch [N] (0,0,0...1,....,2....)
        Affinity= None
        G = None
        if batch is None:
            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, H)
            if self.graph is not None:
                G, Affinity = self.graph(H)

        elif batch.shape[0] == H.shape[0]:
            bag_num = torch.max(batch)+1
            indicator = F.one_hot(batch, bag_num).float()
            A = torch.mul(torch.transpose(A, 1, 0) ,indicator)
            A = torch.softmax(A.sub((1 - indicator).mul(65535)), dim=0)
            M = torch.mm(A.T, H)
            if self.graph is not None:
                G, Affinity = self.graph(H, batch)
                # G = torch.cat([self.GCNblock(H[batch == i])[0].unsqueeze(0) for i in range(0, bag_num)])
                # Affinity = torch.cat([self.GCNblock(H[batch == i])[1].unsqueeze(0) for i in range(0, bag_num)])
                # # Attention = [F.softmax(A.squeeze(0)[batch == i]) for i in range(0, bag_num)]
                M = M + G
        Y_prob = self.classifier(M)
        return Y_prob, Affinity, G

    def calculate_objective(self, X, Y, batch=None):

        Y_prob, Affinity, Graph = self.forward(X, batch=batch)
        if (Y_prob.dim() > 1): ## Batch supported
            Y_hat   = torch.argmax(Y_prob, 1).float() ## [N, ]
            # print(Y)
            neg_log_likelihood = self.criterion(Y_prob, Y)
            Y = Y.float()
            error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
            return neg_log_likelihood, error, torch.argmax(Y_prob, 1), Y
        else:
            Y_hat = torch.argmax(Y_prob).float()
            neg_log_likelihood = self.criterion(Y_prob, Y)
            Y = Y.float()
            error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
            return neg_log_likelihood, error, torch.argmax(Y_prob).item(), Y.item()

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, _= self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        # _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        # Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()

class Agg_GAttention(nn.Module):
    def __init__(self):
        super(Agg_GAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, H, batch=None):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        return A

class Agg_SAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(Agg_SAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.dk = 64

        self.attention_weights = nn.Linear(self.D, self.K)


        # self attention
        self.SA1 = nn.Linear(self.L, self.dk)
        self.SA2 = nn.Linear(self.L, self.dk)
        self.SA = SelfAttention()
        self.dropout = nn.Dropout(dropout)
        #gated attention
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, H, batch=None):
        Q = self.SA1(H)
        K = self.SA2(H)
        H = self.SA(Q, K, H, mask=None, dropout=self.dropout)


        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        return A

class Agg_Attention(nn.Module):
    def __init__(self):
        super(Agg_Attention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Linear(self.L*self.K, 4)
    def forward(self, H, batch=None):
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        return A
# residual = gcn +attention
class GraphMILNet(nn.Module):
    """
    Notes:
        1. Data flow: (B*D1) -> GraphBuilder -> (B*D1, B*B) -> GCN -> (B*D2, B*B) -> Aggregator -> (1*D2, B*B)
        2. For default, the features should be normalized before this module.
    """

    def __init__(self, n_topk):
        super(GraphMILNet, self).__init__()
        self.n_topk = n_topk
        self.gcn = GCNConv(512, 512)



    def forward(self, features, batch):
        bag_num = torch.max(batch) + 1
        norm_features = nn.functional.normalize(features, dim=1)
        correlation_map = torch.mm(norm_features, norm_features.t())  # (batch_size, batch_size)
        adjacent_matrix = self.construct_adjacent_matrix(correlation_map, batch, self.n_topk)
        edge_index = dense_to_sparse(adjacent_matrix)[0]  # [2, E]
        features = self.gcn(features, edge_index)
        agg_features = self.aggregate_features_in_bag(features, batch, bag_num,)
        return agg_features, adjacent_matrix

    def construct_adjacent_matrix(self, correlation_map, indicator, n_topk):
        batch_size = correlation_map.size(0)
        subgraph_mask = torch.eq(*torch.meshgrid(indicator, indicator)).long()  # (batch_size, batch_size)
        # mask connections among nodes from different subgraphs
        adjacent_matrix = torch.mul(correlation_map, subgraph_mask)
        # get indices of non-topk elements
        notopk_indices = adjacent_matrix.topk(batch_size - n_topk, largest=False, dim=1)[1]
        # fill positions of non-topk elements with 0
        adjacent_matrix = adjacent_matrix.scatter(1, notopk_indices, 0)
        return adjacent_matrix

    def aggregate_features_in_bag(self, features, indicator, n_bags, attention_coefficients=None):
        # features should be of shape (batch_size, n_channels)
        # attention_coefficients should be of shape (batch_size, 1)
        indicator_onehot = F.one_hot(indicator, num_classes=n_bags).float()  # (batch_size, n_bags)
        if attention_coefficients is not None:
            attention_coefficients = torch.mul(indicator_onehot,
                                               attention_coefficients.view(-1, 1))  # (batch_size, n_bags)
            attention_coefficients = torch.softmax(
                attention_coefficients.sub((1 - indicator_onehot).mul(65535)), dim=0)
            indicator_onehot = attention_coefficients
        return torch.mm(indicator_onehot.t(), features)  # (n_bags, n_channels)

# class Agg_Residual(AggNet):
#     def __init__(self):
#         super(Agg_Residual, self).__init__()
#         self.L = 512
#         self.D = 128
#         self.K = 1
#         self.knn = 3
#
#         self.criterion = nn.CrossEntropyLoss()
#         self.attention = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh(),
#             nn.Linear(self.D, self.K)
#         )
#         self.attention_V = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh()
#         )
#
#         self.attention_U = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Sigmoid()
#         )
#
#         self.attention_weights = nn.Linear(self.D, self.K)
#
#         # GCN
#         self.graph_builder = KNNGraphBuilder(self.knn)
#         self.gcn = GCNConv(self.L, self.L)
#         self.aggregator = GraphWeightedAvgPooling()
#         self.GCNblock = GraphMILNet(self.graph_builder, self.gcn, self.aggregator)
#         self.classifier = nn.Linear(self.L * self.K, 4)
#
#     def forward(self, H, batch=None):
#         A_V = self.attention_V(H)  # NxD
#         A_U = self.attention_U(H)  # NxD
#         A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
#         A = torch.transpose(A, 1, 0)  # KxN
#         if batch is None:
#             A = F.softmax(A, dim=1)  # softmax over N
#             M = torch.mm(A, H)
#             Y_prob = self.classifier(M)
#         elif batch.shape[0]==H.shape[0]:
#             bag_num = torch.max(batch) + 1
#             ##TODO: support batch as bag size: [N1, N2...NK]
#             ##TODO: implementing with pytorch-scatter
#
#             M = torch.cat([torch.mm(F.softmax(A.squeeze(0)[batch==i].unsqueeze(0), dim=1), H[batch==i])
#                            for i in range(0, bag_num)])
#             G = torch.cat([self.GCNblock(H[batch==i])[0].unsqueeze(0) for i in range(0, bag_num)])
#             Affinity =  torch.cat([self.GCNblock(H[batch==i])[1].unsqueeze(0) for i in range(0, bag_num)])
#             # Attention = [F.softmax(A.squeeze(0)[batch == i]) for i in range(0, bag_num)]
#             Y_prob = self.classifier(M+G)
#
#         return Y_prob, Affinity, G
#
# class Agg_Res_self(AggNet):
#     def __init__(self, dropout=0.1):
#         super(Agg_Res_self, self).__init__()
#         self.L = 512
#         self.D = 128
#         self.K = 1
#         self.knn = 3
#         self.dk = 64
#         self.criterion = nn.CrossEntropyLoss()
#         self.attention_weights = nn.Linear(self.D, self.K)
#
#         self.classifier = nn.Linear(self.L * self.K, 4)
#
#         # self attention
#         self.SA1 = nn.Linear(self.L, self.dk)
#         self.SA2 = nn.Linear(self.L, self.dk)
#         self.SA = SelfAttention()
#         self.dropout = nn.Dropout(dropout)
#         # gated attention
#         self.attention = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh(),
#             nn.Linear(self.D, self.K)
#         )
#         self.attention_V = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh()
#         )
#
#         self.attention_U = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Sigmoid()
#         )
#
#         self.attention_weights = nn.Linear(self.D, self.K)
#
#         # GCN
#         self.graph_builder = KNNGraphBuilder(self.knn)
#         self.gcn = GCNConv(self.L, self.L)
#         self.aggregator = GraphWeightedAvgPooling()
#         self.GCNblock = GraphMILNet(self.graph_builder, self.gcn, self.aggregator)
#         self.classifier = nn.Linear(self.L * self.K, 4)
#
#     def forward(self, H, batch=None):
#         Q = self.SA1(H)
#         K = self.SA2(H)
#         H = self.SA(Q, K, H, mask=None, dropout=self.dropout)
#
#         A_V = self.attention_V(H)  # NxD
#         A_U = self.attention_U(H)  # NxD
#         A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
#         A = torch.transpose(A, 1, 0)  # KxN
#         if batch is None:
#             A = F.softmax(A, dim=1)  # softmax over N
#             M = torch.mm(A, H)
#             Y_prob = self.classifier(M)
#         elif batch.shape[0]==H.shape[0]:
#             bag_num = torch.max(batch) + 1
#             ##TODO: support batch as bag size: [N1, N2...NK]
#             ##TODO: implementing with pytorch-scatter
#
#             M = torch.cat([torch.mm(F.softmax(A.squeeze(0)[batch == i].unsqueeze(0), dim=1), H[batch == i])
#                            for i in range(0, bag_num)])
#             G = torch.cat([self.GCNblock(H[batch == i])[0].unsqueeze(0) for i in range(0, bag_num)])
#             Affinity = torch.cat([self.GCNblock(H[batch == i])[1].unsqueeze(0) for i in range(0, bag_num)])
#             # Attention = [F.softmax(A.squeeze(0)[batch == i]) for i in range(0, bag_num)]
#             Y_prob = self.classifier(M + G)
#
#         return Y_prob, Affinity, G
