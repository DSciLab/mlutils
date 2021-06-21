import inspect
from typing import Any, List
import numpy as np
import torch
from torch import nn
import pickle


class LogitToPreds(object):
    def __init__(self, opt):
        normalization = opt.normalization
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
    def __init__(self, losses, num_pool=0):
        super().__init__()
        self._losses = []
        assert isinstance(losses, list), \
            f'the type of losses should be list, but {type(losses)} got.'
        for item in losses:
            self.append(item['loss_fn'], item['weight'])
        
        if num_pool > 0:
            self.ds_weights = np.array([1 / (2 ** i) for i in range(num_pool)])
            mask = np.array([True if i < num_pool - 1 else False for i in range(num_pool)])
            self.ds_weights[~mask] = 0
            self.ds_weights = self.ds_weights / self.ds_weights.sum()

    def append(self, loss_fn, weight=1.0):
        if (inspect.isfunction(loss_fn) or not inspect.isclass(loss_fn)) \
            and callable(loss_fn):
            self._losses.append({
                'loss_fn': loss_fn,
                'weight': weight
            })

    def _forward(self, inp, gt, *args, **kwargs):
        loss = 0
        for item in self._losses:
            loss += item['loss_fn'](inp, gt, *args, **kwargs) * item['weight']
        return loss

    def _forward_ds(self, inp_list, gt_list, *args, **kwargs):
        loss_list = []
        for i, (inp, gt) in enumerate(zip(inp_list, gt_list)):
            loss_list.append(self._forward(inp, gt, *args, **kwargs) * self.ds_weights[i])
        return torch.stack(loss_list, dim=0).mean(0)

    def forward(self, inp, gt, *args, **kwargs):
        if isinstance(inp, (tuple, list)) and isinstance(gt, (tuple, list)):
            return self._forward_ds(inp, gt, *args, **kwargs)
        elif isinstance(inp, torch.Tensor) and isinstance(gt, torch.Tensor):
            return self._forward(inp, gt, *args, **kwargs)
        else:
            raise ValueError(f'inp and gt data type imcompatibel,'
                             f' type(inp)={type(inp)} and type(gt)={type(gt)}')


class ProgressSign(object):
    def __init__(self) -> None:
        super().__init__()
        self.string = '-\\|/'
        self.index = 0
    
    def __str__(self) -> str:
        char = self.string[self.index]
        self.index = (self.index + 1) % len(self.string)
        return char


def threhold_seg(inp, th=0.5):
    inp[inp>0.5] = 1.
    inp[inp<=0.5] = 0.
    return inp


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def deep_supervised_weight(num_ds: int) -> List:
    """
    from nnUNet
    """
    weights = np.array([1 / (2 ** i) for i in range(num_ds)])

    # we don't use the lowest 2 outputs. Normalize weights so 
    # that they sum to 1
    mask = np.array([True] + [True if i < num_ds - 1 else False 
                                for i in range(1, num_ds)])
    weights[~mask] = 0
    weights = weights / weights.sum()
    
    return weights
