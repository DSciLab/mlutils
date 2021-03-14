import torch
import numpy as np
from sklearn import metrics
from .thirdparty import sklearn_metrics


__all__ = ['Accuracy', 'F1Score', 'AUROC', 'ECE', 'Kappa']


class Metric(object):
    def __init__(self, opt):
        self.num_classes = opt.num_classes
        self.eye = np.eye(self.num_classes)

    @staticmethod
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
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
    def __call__(self, pred, target):
        target = self.de_onehot(target)
        pred = self.de_onehot(pred)
        assert pred.shape == target.shape

        pred = self.to_numpy(pred)
        target = self.to_numpy(target)
        return metrics.accuracy_score(target, pred)

    
class F1Score(Metric):
    def __call__(self, pred, target):
        target = self.de_onehot(target)
        pred = self.de_onehot(pred)
        assert pred.shape == target.shape

        pred = self.to_numpy(pred)
        target = self.to_numpy(target)
        return metrics.f1_score(target, pred,
                                average='micro')


class AUROC(Metric):
    def __call__(self, pred, target):
        target = self.de_onehot(target)
        assert target.shape[0] == pred.shape[0]

        pred = self.to_numpy(pred)
        target = self.to_numpy(target)
        return metrics.roc_auc_score(target, pred,
                                     multi_class='ovo')


class Kappa(Metric):
    def __call__(self, pred, target):
        target = self.de_onehot(target)
        pred = self.de_onehot(pred)
        assert pred.shape == target.shape

        pred = self.to_numpy(pred)
        target = self.to_numpy(target)
        return sklearn_metrics.cohen_kappa_score(
            target, pred, weights='quadratic')


class ECE(Metric):
    def __init__(self):
        super().__init__()
