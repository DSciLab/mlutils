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
        # print('pred.shape', pred.shape)
        # print('target.shape', target.shape)
        if pred.ndim > target.ndim:
            # pred = self.de_onehot(pred)
            pred = self.de_onehot(pred)
        elif pred.ndim < target.ndim:
            target = self.de_onehot(target)
            # target = self.de_onehot(target)
        assert pred.shape == target.shape,\
            f'pred.shape={pred.shape}, target.shape={target.shape}'

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

    # def dice_2d(self, input, target):
    #     return compute_per_channel_dice(input, target)

    # def dice_3d(self, input, target):
    #     return compute_per_channel_dice(input, target)

    def dice(self, test=None, reference=None, confusion_matrix=None,
             nan_for_nonexisting=True, **kwargs):
        """2TP / (2TP + FP + FN)"""

        if confusion_matrix is None:
            confusion_matrix = ConfusionMatrix(test, reference,
                                ignore_index=self.ignore_index)

        tp, fp, tn, fn = confusion_matrix.get_matrix()
        test_empty, test_full, reference_empty, reference_full \
            = confusion_matrix.get_existence()

        if test_empty and reference_empty:
            if nan_for_nonexisting:
                return float("NaN")
            else:
                return 0.

        return float(2. * tp / (2. * tp + fp + fn))

    def __call__(self, logit, target):
        # logit: (B, C, X, Y, Z)
        # target: (B, X, Y, Z)
        # print('logit.shape', logit.shape)
        # print('target.shape', target.shape)
        if not logit.ndim == target.ndim + 1:
            target = target.squeeze(1)
        logit = self.to_numpy(logit)
        target = self.to_numpy(target)
        pred = self.de_onehot(logit)   # to predict
        assert pred.ndim == target.ndim
        # logit: (B, X, Y, Z)
        # target: (B, X, Y, Z)
        pred = expand_as_one_hot(pred, self.num_classes)
        target = expand_as_one_hot(target.astype(np.int),
                                    self.num_classes)
        # logit: (B, C, X, Y, Z)
        # target: (B, C, X, Y, Z)

        return self.dice(pred, target)


class ConfusionMatrix:
    def __init__(self, test=None, reference=None, ignore_index=None):
        assert ignore_index is None or isinstance(ignore_index, int)
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.ignore_index = ignore_index

        assert reference.shape == test.shape
        if self.ignore_index is not None:
            self.gether_indices = list(range(reference.shape[1]))
            self.gether_indices.remove(self.ignore_index)
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):
        if self.ignore_index is None:
            self.test = test
        else:
            self.test = np.take(
                test, indices=self.gether_indices, axis=1)
        self.reset()

    def set_reference(self, reference):
        if self.ignore_index is None:
            self.reference = reference
        else:
            self.reference = np.take(
                reference, indices=self.gether_indices, axis=1)
        self.reset()

    def reset(self):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):
        if self.test is None or self.reference is None:
            raise ValueError('\'test\' and \'reference\' must both be set '
                             'to compute confusion matrix.')

        assert self.test.shape == self.reference.shape, \
               f'Shape mismatch: {self.test.shape} and {self.reference.shape}.'

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())

        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):
        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):
        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):
        for case in (self.test_empty, self.test_full,
                     self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, \
               self.reference_empty, self.reference_full


# def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
#     """
#     Computes DiceCoefficient as defined in 
#         https://arxiv.org/abs/1606.04797 given  a multi channel 
#         input and target.
#     Assumes the input is a normalized probability, e.g. 
#         a result of Sigmoid or Softmax function.

#     Args:
#          input (torch.Tensor): NxCxSpatial input tensor
#          target (torch.Tensor): NxCxSpatial target tensor
#          epsilon (float): prevents division by zero
#          weight (torch.Tensor): Cx1 tensor of weight per channel/class
#     """

#     # input and target shapes must match
#     assert input.shape == target.shape, "'input' and 'target' must have the same shape"

#     input = flatten(input)
#     target = flatten(target)
#     target = target.astype(np.float)

#     # compute per channel Dice Coefficient
#     intersect = (input * target).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect

#     # here we can use standard dice (input + target).sum(-1)
#     denominator = input.sum(-1) + target.sum(-1)
#     return 2 * (intersect / denominator.clip(min=epsilon)).mean()


# def flatten(array):
#     """Flattens a given tensor such that the channel axis is first.
#     The shapes are transformed as follows:
#        (N, C, D, H, W) -> (C, N * D * H * W)
#     """
#     # number of channels
#     C = array.shape[1]
#     # new axis order
#     axis_order = (1, 0) + tuple(range(2, array.ndim))
#     # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
#     # transposed = tensor.permute(axis_order)
#     transposed = np.transpose(array, (axis_order))
#     # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
#     return transposed.reshape(C, -1)


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
