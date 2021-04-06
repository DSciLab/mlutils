import torch
from torch import nn
import numpy as np
from sklearn import metrics
from .thirdparty import sklearn_metrics


__all__ = ['Accuracy', 'F1Score', 'AUROC', 'ECE', 'Kappa']


def threhold(inp, th=0.5):
    inp_ = np.copy(inp)
    inp_[inp_>th] = 1.
    inp_[inp_<=th] = 0.
    return inp_


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

    @property
    def name(self):
        return self.__class__.__name__

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
        pred = self.de_onehot(pred)
        if pred.ndim != target.ndim:
            target = self.de_onehot(target)
        assert pred.shape == target.shape

        pred = self.to_numpy(pred)
        pred = threhold(pred)
        target = self.to_numpy(target)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        return metrics.accuracy_score(target, pred)


class F1Score(Metric):
    def __call__(self, pred, target):
        target = self.de_onehot(target)
        pred = self.de_onehot(pred)
        assert pred.shape == target.shape

        pred = self.to_numpy(pred)
        pred = threhold(pred)
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
        pred = threhold(pred)
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
        self.ignore_index = opt.get('ignore_index', None)

    def dice_2d(self, input, target):
        if input.ndim == target.ndim + 1:
            target = expand_as_one_hot(target, self.num_classes)

        return compute_per_channel_dice(input, target)

    def dice_3d(self, input, target):
        if input.ndim == target.ndim + 1:
            target = expand_as_one_hot(target, self.num_classes)

        return compute_per_channel_dice(input, target)

    def __call__(self, input, target):
        input = self.to_numpy(input)
        input = threhold(input)
        target = self.to_numpy(target)

        if input.ndim == 4:
            # N * X * Y
            return self.dice_2d(input, target)
        elif input.ndim == 5:
            # N * X * Y * Z
            return self.dice_3d(input, target)
        else:
            raise RuntimeError(
                f'The shape of target is {target.shape}.')


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    # TODO ignore index
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
    assert input.shape == target.shape, "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.astype(np.float)

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1)
    denominator = input.sum(-1) + target.sum(-1)
    return 2 * (intersect / denominator.clip(min=epsilon)).mean()


def flatten(array):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = array.shape[1]
    # new axis order
    axis_order = (1, 0) + tuple(range(2, array.ndim))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    # transposed = tensor.permute(axis_order)
    transposed = np.transpose(array, (axis_order))
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


def expand_as_one_hot(input, C):
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
    # assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    # input = input.unsqueeze(1)
    input = np.expand_dims(input, axis=1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.shape)
    shape[1] = C

    # scatter to get the one-hot tensor
    # return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
    result = np.zeros(shape)
    np.put_along_axis(result, input, 1, axis=1)
    return result

'''
class Dice(Metric):
    def __init__(self, opt):
        super().__init__(opt)
        normalization = opt.get('normalization', 'sigmoid')
        assert normalization in ['sigmoid', 'softmax', 'none']

        self.num_classes = opt.num_classes

    def dice_2d(self, input, target):
        if input.dim() == target.dim() + 1:
            target = expand_as_one_hot(target, self.num_classes)

        return compute_per_channel_dice(input, target)

    def dice_3d(self, input, target):
        if input.dim() == target.dim() + 1:
            target = expand_as_one_hot(target, self.num_classes)

        return compute_per_channel_dice(input, target)

    def __call__(self, input, target):
        input[input<=0.5] = 0.0
        input[input>0.5] = 1.0

        if target.ndim == 3:
            # N * X * Y
            return self.dice_2d(input, target)
        elif target.ndim == 4:
            # N * X * Y * Z
            return self.dice_3d(input, target)
        else:
            raise RuntimeError(
                f'The shape of target is {target.shape}.')


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
    # assert input.dim() == 4

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
'''
