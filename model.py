import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from GraphMIL.GraphBuilder.KNNGraphBuilder import KNNGraphBuilder
from GraphMIL.GCN.GCNConv import GCNConv
from GraphMIL.Aggregator.GraphWeightedAvgPooling import GraphWeightedAvgPooling
from torch_geometric.utils import dense_to_sparse
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.criterion = nn.CrossEntropyLoss()
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Linear(self.L*self.K, 4)
        model_resnet50 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool

    def forward(self, x):
        x = x.squeeze(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        H = x.view(x.size(0), -1)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        return Y_prob, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        # _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        # Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()

    def calculate_objective(self, X, Y):
        Y_prob, A = self.forward(X)
        Y_hat = torch.argmax(Y_prob).float()
        neg_log_likelihood = self.criterion(Y_prob, Y)
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, error, A, torch.argmax(Y_prob).item(), Y.item()

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
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
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.L * self.K, 1),
        #     nn.Sigmoid()
        # )
        model_resnet50 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x):
        x = x.squeeze(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        H = x.view(x.size(0), -1)

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        # Y_hat = torch.argmax(Y_prob).float()
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, A

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
    #
    #     return error, Y_hat
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, _= self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        # _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        # Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()

    def calculate_objective(self, X, Y):

        Y_prob, A = self.forward(X)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        Y_hat = torch.argmax(Y_prob).float()
        neg_log_likelihood = self.criterion(Y_prob, Y)
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, error, A, torch.argmax(Y_prob).item(), Y.item()

class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Linear(self.L*self.K, 4)
        model_resnet50 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x):
        if len(x.shape)>4:
            x = x.squeeze(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        H = x.view(x.size(0), -1)


        Y_prob = self.classifier(H)
        # Y_hat = torch.argmax(Y_prob).float()
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob

    def calculate_objective(self, X, Y):

        Y_prob = self.forward(X)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        Y_hat = torch.argmax(Y_prob).float()
        neg_log_likelihood = self.criterion(Y_prob, Y)
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, error, None, torch.argmax(Y_prob,dim=1).cpu(), Y.cpu()

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob= self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()

class Res18_SAFS(nn.Module):
    def __init__(self, dropout=0.1):
        super(Res18_SAFS, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.dk = 64
        self.criterion = nn.CrossEntropyLoss()
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Linear(self.L*self.K, 4)

        model_resnet50 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
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

    def forward(self, x):
        if len(x.shape)>4:
            x = x.squeeze(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        H = x.view(x.size(0), -1)

        Q = self.SA1(H)
        K = self.SA2(H)
        H = self.SA(Q, K, H, mask=None, dropout=self.dropout)


        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        # Y_hat = torch.argmax(Y_prob).float()
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob



    def calculate_objective(self, X, Y):

        Y_prob = self.forward(X)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        Y_hat = torch.argmax(Y_prob).float()
        neg_log_likelihood = self.criterion(Y_prob, Y)
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, error, None, torch.argmax(Y_prob,dim=1).cpu(), Y.cpu()

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob= self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        # _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        # Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()


class SelfAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)

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
        x = x.squeeze(0)
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

class Agg_GAttention(nn.Module):
    def __init__(self):
        super(Agg_GAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.criterion = nn.CrossEntropyLoss()
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
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        ## batch: map each instance to corresponding bag in a mini-batch [N] (0,0,0...1,....,2....)
        if batch is None:
            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, H)
            Y_prob = self.classifier(M)
        elif batch.shape[0]==H.shape[0]:
            bag_num = torch.max(batch)[0].item()
            ##TODO: support batch as bag size: [N1, N2...NK]
            ##TODO: implementing with pytorch-scatter
            A = [F.softmax(A[torch.nonzero(batch==i)], dim=1) for i in range(0, bag_num)]
            M = [torch.mm(A[torch.nonzero(batch==i)], H[torch.nonzero(batch==i)]) for i in range(0, bag_num)]
            Y_prob = torch.stack([self.classifier(M[i]) for i in range(0, bag_num)]) # [bg_num, 4]
        # Y_hat = torch.argmax(Y_prob).float()
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, A

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
    #
    #     return error, Y_hat
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, _= self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        # _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        # Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()

    def calculate_objective(self, X, Y, batch=None):

        Y_prob, A = self.forward(X, batch=batch)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        if (Y_prob.dim() > 1): ## Batch supported
            Y_hat   = torch.argmax(Y_prob, 1).float() ## [N, ]
            neg_log_likelihood = self.criterion(Y_prob, Y)
            Y = Y.float()
            error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
            return neg_log_likelihood, error, A, torch.argmax(Y_prob, 1), Y
        else:
            Y_hat = torch.argmax(Y_prob).float()
            neg_log_likelihood = self.criterion(Y_prob, Y)
            Y = Y.float()
            error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
            return neg_log_likelihood, error, A, torch.argmax(Y_prob).item(), Y.item()

class Agg_SAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(Agg_SAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.dk = 64
        self.criterion = nn.CrossEntropyLoss()
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Linear(self.L*self.K, 4)

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

    def forward(self, H):
        Q = self.SA1(H)
        K = self.SA2(H)
        H = self.SA(Q, K, H, mask=None, dropout=self.dropout)


        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        # Y_hat = torch.argmax(Y_prob).float()
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob



    def calculate_objective(self, X, Y):

        Y_prob = self.forward(X)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        Y_hat = torch.argmax(Y_prob).float()
        neg_log_likelihood = self.criterion(Y_prob, Y)
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, error, None, torch.argmax(Y_prob,dim=1).cpu(), Y.cpu()

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob= self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        # _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        # Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()

class Agg_Attention(nn.Module):
    def __init__(self):
        super(Agg_Attention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.criterion = nn.CrossEntropyLoss()
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
    def forward(self, H):
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        # Y_hat = torch.argmax(Y_prob).float()
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, A

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
    #
    #     return error, Y_hat
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, _= self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        # _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        # Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()

    def calculate_objective(self, X, Y):

        Y_prob, A = self.forward(X)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        Y_hat = torch.argmax(Y_prob).float()
        neg_log_likelihood = self.criterion(Y_prob, Y)
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, error, A, torch.argmax(Y_prob).item(), Y.item()


class GraphMILNet(nn.Module):
    """
    Notes:
        1. Data flow: (B*D1) -> GraphBuilder -> (B*D1, B*B) -> GCN -> (B*D2, B*B) -> Aggregator -> (1*D2, B*B)
        2. For default, the features should be normalized before this module.
    """

    def __init__(self, graph_builder, gcn, aggregator):
        super(GraphMILNet, self).__init__()
        self.graph_builder = graph_builder
        self.gcn = gcn
        self.aggregator = aggregator

    def forward(self, x):
        x, A = self.graph_builder(x)
        edge_index = dense_to_sparse(A)[0]  # [2, E]
        x = self.gcn(x, edge_index)
        return self.aggregator(x, A)
class Agg_Residual(nn.Module):
    def __init__(self):
        super(Agg_Residual, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.knn = 3

        self.criterion = nn.CrossEntropyLoss()
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

        # GCN
        self.graph_builder = KNNGraphBuilder(self.knn)
        self.gcn = GCNConv(self.L, self.L)
        self.aggregator = GraphWeightedAvgPooling()
        self.GCNblock = GraphMILNet(self.graph_builder, self.gcn, self.aggregator)
        self.classifier = nn.Linear(self.L * self.K, 4)

    def forward(self, H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        G = self.GCNblock(H)
        M += G.unsqueeze(0)
        Y_prob = self.classifier(M)
        # Y_hat = torch.argmax(Y_prob).float()
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, A

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
    #
    #     return error, Y_hat
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, _ = self.forward(X)
        # Y_hat = torch.argmax(Y_prob, dim=1).float().max()
        Y_hat = torch.argmax(Y_prob, dim=1).float()
        # _, counts = torch.unique(Y_hat, sorted=True, return_counts=True)
        # Y_hat = torch.argmax(counts)
        # error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return Y_hat.cpu(), Y.cpu()

    def calculate_objective(self, X, Y):
        Y_prob, A = self.forward(X)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        Y_hat = torch.argmax(Y_prob).float()
        neg_log_likelihood = self.criterion(Y_prob, Y)
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, error, A, torch.argmax(Y_prob).item(), Y.item()
