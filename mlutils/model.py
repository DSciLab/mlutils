import torch
from torch import nn


class EMAModel(nn.Module):
    def __init__(self, opt, init_model):
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)
        self.alpha = opt.ema_alpha
        self.shadow_model = init_model

    @property
    def device(self):
        return self._dummy_param.device

    def update(self, model):
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for shadow_param, param in zip(self.shadow_model.parameters(), model.parameters()):
            shadow_param.data.mul_(alpha).add_(1 - alpha, param.data.to(self.device))

    def forward(self, x):
        self.global_step += 1
        return self.shadow_model(x)
