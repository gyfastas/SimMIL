import torch
import torch.nn as nn

class MeanTeacher(nn.Module):
    def __init__(self, m, model_factory, *args, **kwargs):

        super().__init__()
        self.m = m
        self.model = model_factory(**kwargs)
        self.ema_model = model_factory(**kwargs)
        
        self.single_head = False
        self.use_ema = False

        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def eval(self):
        self.single_head = True
        return self.train(False)

    @torch.no_grad()
    def momentum_update_ema_model(self, global_step=None):
        """
        Momentum update of the mean teacher
        """
        m = min(1 - 1 / (global_step + 1), self.m) if global_step else self.m
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def forward(self, imgs, use_ema=None):
        if use_ema is not None:
            self.use_ema = use_ema
        if self.use_ema:
            return self.ema_model(imgs)[0] if self.single_head else self.ema_model(imgs)
        else:
            return self.model(imgs)[0] if self.single_head else self.model(imgs)