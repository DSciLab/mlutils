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
        self.string = None

    def reset(self):
        self.data_sum = None
        self.cnt = 0
        self.max = -np.Inf
        self.min = np.Inf
        self.avg = None
        self.latest = None

    def step(self):
        if self.avg is not None:
            if self.avg > self.max:
                self.max = self.avg
            if self.avg < self.min:
                self.min = self.avg
        self.latest = self.avg

    def zero(self):
        self.cnt = 0
        self.data_sum = 0
        self.avg = 0

    def append(self, data, cnt=1):
        self.cnt += cnt
        data = data.mean()
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if self.data_sum is None:
            self.data_sum = data
        else:
            self.data_sum += data

        self.avg = self.data_sum / self.cnt

    def __str__(self):
        string = f'[{self.name}] '
        if self.avg is not None:
            string += f'latest: {self.latest:.4f} | '
            string += f'avg: {self.avg:.4f} | '
            string += f'max: {self.max:.4f} | '
            string += f'min: {self.min:.4f} | '
            string += f'cnt: {self.cnt} |'
        return string
