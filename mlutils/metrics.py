import torch
import bisect
import math
from torch import nn
import numpy as np
from sklearn import metrics
from .thirdparty import sklearn_metrics


__all__ = ['Accuracy', 'F1Score', 'AUROC', 'ECE', 'Kappa', 'Dice', 'Dice2d']


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
        return metrics.f1_score(
            target, pred,
            average='micro'
        )


class Recall(Metric):
    def __call__(self, pred, target):
        target = self.de_onehot(target)
        pred = self.de_onehot(pred)
        assert pred.shape == target.shape

        pred = self.to_numpy(pred)
        pred = threhold(pred)
        target = self.to_numpy(target)
        return metrics.recall_score(
            target, pred,
            average='binary',
            zero_division=0
        )

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
    def __init__(self, opt):
        super().__init__(opt)

    def gce(
        self,
        labels,
        probs,
        binning_scheme,
        max_prob,
        class_conditional,
        norm,
        num_bins=30,
        threshold=0.0,
        datapoints_per_bin=None
    ):
        metric = GeneralCalibrationError(
            num_bins=num_bins,
            binning_scheme=binning_scheme,
            class_conditional=class_conditional,
            max_prob=max_prob,
            norm=norm,
            threshold=threshold,
            datapoints_per_bin=datapoints_per_bin
        )
        metric.update_state(labels, probs)
        return metric.result()

    def __call__(self, probs, labels, num_bins=30):
        return self.gce(
            labels,
            probs,
            binning_scheme='even',
            max_prob=True,
            class_conditional=False,
            norm='l1',
            num_bins=num_bins
        )


class Dice2d(Metric):
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
        # logit: (B, C, X, Y)
        # target: (B, X, Y)
        # print('logit.shape', logit.shape)
        # print('target.shape', target.shape)
        if not logit.ndim == target.ndim + 1:
            target = target.squeeze(1)
        logit = self.to_numpy(logit)
        target = self.to_numpy(target)
        pred = self.de_onehot(logit)   # to predict
        assert pred.ndim == target.ndim
        # logit: (B, X, Y)
        # target: (B, X, Y)
        pred = expand_as_one_hot(pred, self.num_classes)
        target = expand_as_one_hot(
            target.astype(np.int), self.num_classes
        )
        # logit: (B, C, X, Y)
        # target: (B, C, X, Y)

        return self.dice(pred, target)


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


def to_bins(values, bin_lower_bounds):
  """Use binary search to find the appropriate bin for each value."""
  return np.array([
      bisect.bisect_left(bin_lower_bounds, value)-1 for value in values])


def one_hot_encode(labels, num_classes=None):
  """One hot encoder for turning a vector of labels into a OHE matrix."""
  if num_classes is None:
    num_classes = len(np.unique(labels))
  return np.eye(num_classes)[labels]


def mean(inputs):
  """Be able to take the mean of an empty array without hitting NANs."""
  # pylint disable necessary for numpy and pandas
  if len(inputs) == 0:  # pylint: disable=g-explicit-length-test
    return 0
  else:
    return np.mean(inputs)


def get_adaptive_bins(predictions, num_bins):
  """Returns lower bounds for binning an equal number of datapoints per bin."""

  sorted_predictions = np.sort(predictions)
  # Compute switching point to handle the remainder when allocating the number
  # of examples equally across bins. Up to the switching index, bins use
  # ceiling to round up; after the switching index, bins use floor.
  examples_per_bin = sorted_predictions.shape[0] / float(num_bins)
  switching_index = int(math.floor((examples_per_bin % 1) * num_bins))
  indices = []
  index = 0
  while index < sorted_predictions.shape[0]:
    indices.append(index)
    if index < switching_index:
      index += int(math.ceil(examples_per_bin))
    else:
      index += int(math.floor(examples_per_bin))
  indices = np.array(indices)
  bins = sorted_predictions[indices.astype(np.int32)]
  return bins


def binary_converter(probs):
  """Converts a binary probability vector into a matrix."""
  return np.array([[1-p, p] for p in probs])


class GeneralCalibrationError():
  """Implements the space of calibration errors, General Calibration Error.
  This implementation of General Calibration Error can be class-conditional,
  adaptively binned, thresholded, focus on the maximum or top labels, and use
  the l1 or l2 norm. Can function as ECE, SCE, RMSCE, and more. For
  definitions of most of these terms, see [1].
  To implement Expected Calibration Error [2]:
  ECE = GeneralCalibrationError(binning_scheme='even', class_conditional=False,
    max_prob=True, error='l1')
  To implement Static Calibration Error [1]:
  SCE = GeneralCalibrationError(binning_scheme='even', class_conditional=False,
    max_prob=False, error='l1')
  To implement Root Mean Squared Calibration Error [3]:
  RMSCE = GeneralCalibrationError(binning_scheme='adaptive',
  class_conditional=True, max_prob=False, error='l2', datapoints_per_bin=100)
  To implement Adaptive Calibration Error [1]:
  ACE = GeneralCalibrationError(binning_scheme='adaptive',
  class_conditional=True, max_prob=False, error='l1')
  To implement Thresholded Adaptive Calibration Error [1]:
  TACE = GeneralCalibrationError(binning_scheme='adaptive',
  class_conditional=True, max_prob=False, error='l1', threshold=0.01)
  ### References
  [1] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel,
  and Dustin Tran. "Measuring Calibration in Deep Learning." In Proceedings of
  the IEEE Conference on Computer Vision and Pattern Recognition Workshops,
  pp. 38-41. 2019.
  https://arxiv.org/abs/1904.01685
  [2] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
  "Obtaining well calibrated probabilities using bayesian binning."
  Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/
  [3] Hendrycks, Dan, Mantas Mazeika, and Thomas Dietterich.
  "Deep anomaly detection with outlier exposure."
  arXiv preprint arXiv:1812.04606 (2018).
  https://arxiv.org/pdf/1812.04606.pdf
  Attributes:
    binning_scheme: String, either 'even' (for even spacing) or 'adaptive'
      (for an equal number of datapoints in each bin).
    max_prob: Boolean, 'True' to measure calibration only on the maximum
      prediction for each datapoint, 'False' to look at all predictions.
    class_conditional: Boolean, 'False' for the case where predictions from
      different classes are binned together, 'True' for binned separately.
    norm: String, apply 'l1' or 'l2' norm to the calibration error.
    num_bins: Integer, number of bins of confidence scores to use.
    threshold: Float, only look at probabilities above a certain value.
    datapoints_per_bin: Int, number of datapoints in each adaptive bin. This
      is a second option when binning adaptively - you can use either num_bins
      or this method to determine the bin size.
    distribution: String, data distribution this metric is measuring, whether
      train, test, out-of-distribution, or the user's choice.
    accuracies: Vector, accuracy within each bin.
    confidences: Vector, mean confidence within each bin.
    calibration_error: Float, computed calibration error.
    calibration_errors: Vector, difference between accuracies and confidences.
  """

  def __init__(self,
               binning_scheme,
               max_prob,
               class_conditional,
               norm,
               num_bins=30,
               threshold=0.0,
               datapoints_per_bin=None,
               distribution=None):
    self.binning_scheme = binning_scheme
    self.max_prob = max_prob
    self.class_conditional = class_conditional
    self.norm = norm
    self.num_bins = num_bins
    self.threshold = threshold
    self.datapoints_per_bin = datapoints_per_bin
    self.distribution = distribution
    self.accuracies = None
    self.confidences = None
    self.calibration_error = None
    self.calibration_errors = None

  def get_calibration_error(self, probs, labels, bin_lower_bounds, norm,
                            num_bins):
    """Given a binning scheme, returns sum weighted calibration error."""
    bins = to_bins(probs, bin_lower_bounds)
    self.confidences = np.nan_to_num(
        np.array([mean(probs[bins == i]) for i in range(num_bins)]))
    counts = np.array(
        [len(probs[bins == i]) for i in range(num_bins)])
    self.accuracies = np.nan_to_num(
        np.array([mean(labels[bins == i]) for i in range(num_bins)]))
    self.calibration_errors = self.accuracies-self.confidences
    weighting = counts / float(len(probs.flatten()))
    weighted_calibration_error = self.calibration_errors * weighting
    if norm == 'l1':
      return np.sum(np.abs(weighted_calibration_error))
    else:
      return np.sum(np.square(weighted_calibration_error))

  def update_state(self, labels, probs):
    """Updates the value of the General Calibration Error."""

    # if self.calibration_error is not None and

    probs = np.array(probs)
    labels = np.array(labels)
    if probs.ndim == 2:

      num_classes = probs.shape[1]
      if num_classes == 1:
        probs = probs[:, 0]
        probs = binary_converter(probs)
        num_classes = 2
    elif probs.ndim == 1:
      # Cover binary case
      probs = binary_converter(probs)
      num_classes = 2
    else:
      raise ValueError('Probs must have 1 or 2 dimensions.')

    # Convert the labels vector into a one-hot-encoded matrix.
    labels_matrix = one_hot_encode(labels, probs.shape[1])

    if self.datapoints_per_bin is not None:
      self.num_bins = int(len(probs)/self.datapoints_per_bin)
      if self.binning_scheme != 'adaptive':
        raise ValueError(
            "To set datapoints_per_bin, binning_scheme must be 'adaptive'.")

    if self.binning_scheme == 'even':
      bin_lower_bounds = [float(i)/self.num_bins for i in range(self.num_bins)]

    # When class_conditional is False, different classes are conflated.
    if not self.class_conditional:
      if self.max_prob:
        labels_matrix = labels_matrix[
            range(len(probs)), np.argmax(probs, axis=1)]
        probs = probs[range(len(probs)), np.argmax(probs, axis=1)]
      labels_matrix = labels_matrix[probs > self.threshold]
      probs = probs[probs > self.threshold]
      if self.binning_scheme == 'adaptive':
        bin_lower_bounds = get_adaptive_bins(probs, self.num_bins)
      calibration_error = self.get_calibration_error(
          probs.flatten(), labels_matrix.flatten(), bin_lower_bounds, self.norm,
          self.num_bins)

    # If class_conditional is true, predictions from different classes are
    # binned separately.
    else:
      # Initialize list for class calibration errors.
      class_calibration_error_list = []
      for j in range(num_classes):
        if not self.max_prob:
          probs_slice = probs[:, j]
          labels = labels_matrix[:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          if self.binning_scheme == 'adaptive':
            bin_lower_bounds = get_adaptive_bins(probs_slice, self.num_bins)
          calibration_error = self.get_calibration_error(
              probs_slice, labels, bin_lower_bounds, self.norm, self.num_bins)
          class_calibration_error_list.append(calibration_error/num_classes)
        else:
          # In the case where we use all datapoints,
          # max label has to be applied before class splitting.
          labels = labels_matrix[np.argmax(probs, axis=1) == j][:, j]
          probs_slice = probs[np.argmax(probs, axis=1) == j][:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          if self.binning_scheme == 'adaptive':
            bin_lower_bounds = get_adaptive_bins(probs_slice, self.num_bins)
          calibration_error = self.get_calibration_error(
              probs_slice, labels, bin_lower_bounds, self.norm, self.num_bins)
          class_calibration_error_list.append(calibration_error/num_classes)
      calibration_error = np.sum(class_calibration_error_list)

    if self.norm == 'l2':
      calibration_error = np.sqrt(calibration_error)

    self.calibration_error = calibration_error

  def result(self):
    return self.calibration_error

  def reset_state(self):
    self.calibration_error = None
