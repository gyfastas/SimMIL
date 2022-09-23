import torch.nn as nn
import torch
import torch.nn.functional as F



def prob_max(logits, pos_class):
    """
    Args:
        logits: [N, B, C]
        pos_class: int
    Returns:
        output: [N, 2]
    convert to instance level binary classification => take max prob for each logit
    """
    logits = logits.softmax(-1).max(1)[0] # [N, B, C] => [N, C]
    pos_logit = logits.clone()[:, pos_class]
    logits[:, pos_class] = 0
    neg_logit = logits.max(1)[0]
    return torch.stack([neg_logit, pos_logit], 1)

def prob_mean(logits, pos_class):
    logits = logits.softmax(-1).mean(1) # [N, B, C] => [N, C] 
    pos_logit = logits.clone()[:, pos_class]
    logits[:, pos_class] = 0
    neg_logit = logits.max(1)[0]
    return torch.stack([neg_logit, pos_logit], 1)

# def prob_mean(logits, pos_class):
#     pos_logit = logits.clone()[:, :, pos_class] # [N, B]
#     logits[:, :, pos_class] = 0
#     neg_logit = logits.max(2)[0] # [N, B]
#     logits = torch.stack([neg_logit, pos_logit], 2) # [N, B, 2]
#     logits = logits.softmax(-1).mean(1) # [N, B, 2] => [N, 2] 
#     return logits

def major_vote(logits, pos_class):
    cn = logits.shape[-1]
    logits = F.one_hot(logits.argmax(2), cn).to(torch.float) # [N, B, C] => [N, B, C]
    logits = logits.sum(1) # [N, B, C] => [N, C]
    pos_logit = logits.clone()[:, pos_class]
    logits[:, pos_class] = 0
    neg_logit = logits.max(1)[0]
    return torch.stack([neg_logit, pos_logit], 1)
    
def aggregate_factory(agg_type):
    aggregate_dict = dict(prob_max=prob_max, prob_mean=prob_mean, major_vote=major_vote)
    return aggregate_dict[agg_type]

class InsMILModel(nn.Module):
    def __init__(self, backbone_factory, ins_aggregation, class_num, pos_class):
        super().__init__()
        self.backbone = backbone_factory(num_classes=class_num) 
        self.aggregator = aggregate_factory(ins_aggregation)
        self.pos_class = pos_class

    def to_mil_pred(self, logits):
        with torch.no_grad():
            # convert to Binary Classfication Prediction
            pos_logit = logits[:, self.pos_class].clone().view(-1)
            logits[:, self.pos_class] = 0.0

            neg_logit = logits.max(1)[0].view(-1) #
            pred = torch.stack([neg_logit, pos_logit], 1).softmax(1)
            return pred

    def forward(self, x):
        # x: [N, B, C, H, W]
        n,b,_,_,_ = x.shape
        logits = self.backbone(x.view(n*b, *x.shape[2:])) # feats: [N*B, C]
        logits = logits.view(n, b, -1)
        return self.aggregator(logits, self.pos_class)