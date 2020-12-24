import numpy as np
import torch


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.data_sum = None
        self.cnt = 0
        self.max = -np.Inf
        self.min = np.Inf
        self.avg = None
        self.latest = None

    def append(self, data, cnt=1):
        self.cnt += cnt
        data = data.mean()
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        self.latest = data
        if self.data_sum is None:
            self.data_sum = data
        else:
            self.data_sum += data

        self.avg = self.data_sum / self.cnt
        if data > self.max:
            self.max = data
        if data < self.min:
            self.min = data

    def __str__(self):
        string = f'[{self.name}] '
        string += f'latest: {self.latest:.4f} | '
        string += f'avg: {self.avg:.4f} | '
        string += f'max: {self.max:.4f} | '
        string += f'min: {self.min:.4f} | '
        string += f'cnt: {self.cnt:.4f} |'

        return string
