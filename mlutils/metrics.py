from numpy.lib.arraysetops import isin
import torch
import numpy as np
import sklearn


__all__ = ['Accuracy', 'F1Score', 'AUROC', 'ECE', 'Kappa']


class Metric(object):
    def __init__(self, opt):
        self.num_classes = opt.num_classes
        self.eye = np.eye(self.num_classes)

    @staticmethod
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.numpy()
        else:
            return data

    def onehot(self, labels):
        if labels.ndim == 2:
            return labels
        else:
            return self.eye[labels]

    @staticmethod
    def de_onehot(label_matrix):
        if label_matrix.ndim == 1:
            return label_matrix
        else:
            return label_matrix.argmax(axis=1)

    def __call__(self):
        raise NotImplementedError


class Accuracy(Metric):
    def __init__(self):
        super().__init__()

    
class F1Score(Metric):
    def __init__(self):
        super().__init__()


class AUROC(Metric):
    def __init__(self):
        super().__init__()


class ECE(Metric):
    def __init__(self):
        super().__init__()

class Kappa(Metric):
    def __init__(self):
        super().__init__()
