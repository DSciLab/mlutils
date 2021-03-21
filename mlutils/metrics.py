import torch
from torch import nn
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


class Dice(Metric):
    def __init__(self, opt):
        super().__init__(opt)
        normalization = opt.get('normalization', 'sigmoid')
        assert normalization in ['sigmoid', 'softmax', 'none']

        self.num_classes = opt.num_classes
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def __call__(self, input, target):
        input = self.normalization(input)
        if input.dim() == target.dim() + 1:
            target = expand_as_one_hot(target, self.num_classes)

        return compute_per_channel_dice(input, target)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in 
        https://arxiv.org/abs/1606.04797 given  a multi channel 
        input and target.
    Assumes the input is a normalized probability, e.g. 
        a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension
    # (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets
        converted to its corresponding one-hot vector. It is assumed that
        the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
