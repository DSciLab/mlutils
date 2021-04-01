import inspect
import torch
from torch import nn


class LogitToPreds(object):
    def __init__(self, opt):
        normalization = opt.get('normalization', 'sigmoid')
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def __call__(self, logit):
        assert isinstance(logit, torch.Tensor), \
            f'Type error, the type of logit is expected to be ' + \
            f'torch.Tensor, but {type(logit)} got.'

        return self.normalization(logit)


class MultiLoss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self._losses = []
        assert isinstance(losses, list), \
            f'the type of losses should be list, but {type(losses)} got.'
        for item in losses:
            self.append(item['loss_fn'], item['weight'])

    def append(self, loss_fn, weight=1.0):
        if (inspect.isfunction(loss_fn) or not inspect.isclass(loss_fn)) \
            and callable(loss_fn):
            self._losses.append({
                'loss_fn': loss_fn,
                'weight': weight
            })

    def forward(self, inp, gt):
        loss = 0
        for item in self._losses:
            loss += item['loss_fn'](inp, gt) * item['weight']
        return loss


def threhold_seg(inp, th=0.5):
    inp[inp>0.5] = 1.
    inp[inp<=0.5] = 0.
    return inp
