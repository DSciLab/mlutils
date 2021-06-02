from typing import Callable, Optional, Union, Tuple, List
import torch
from torch import nn
from cfg import Opts
from torch import Tensor
from torch.nn import functional as F
from mlutils import LogitToPreds


EPS = 1.0e-8


__all__ = ['IOULoss', 'GDiceLoss', 'SoftDiceLoss',
           'CrossEntropyLoss', 'BCELossWithLogits',
           'GDiceCELoss', 'GDiceBCELoss', 'SoftDiceCELoss',
           'SoftDiceBCELoss', 'DeepSupervisedLoss',
           'LossPicker']


def softmax_helper(inp: Tensor) -> Tensor:
    return F.softmax(inp, 1)


def onehot(inp: Tensor, num_classes: int,
           with_channel: Optional[bool]=False) -> Tensor:
    if not with_channel:
        inp = inp.unsqueeze(1)

    output_shape = list(inp.shape)
    output_shape[1] = num_classes

    output = torch.zeros(output_shape, dtype=torch.float).to(inp.device)
    output.scatter_(1, inp.type(torch.int64), 1)
    return output


def flatten(inp: Tensor, with_class: Optional[bool]=False) -> Tensor:
    """
    :param inp: input tensor with shape (B, C, Spatial_shape)
    :param with_class: flatten the tensor with C dim
    :return: if with_class is True, return a tensor woth shape of 
             (B, C, prod(spatial_shape)), if with_class is False,
             return a tensor with shape of (B, C * prod(spatial_shape))
    """
    if with_class:
        B = inp.size(0)
        C = inp.size(1)
        inp = inp.view(B, C, -1)
    else:
        B = inp.size(0)
        inp = inp.view(B, -1)
    return inp


def flatten_with_class(inp: Tensor) -> Tensor:
    """
    :param inp: input tensor, the expected shape is (B, C, spatial_shape)
    :return: a tentor with shape (C, B * prod(spatial_shape))
    """
    inp = inp.permute(1, 0, *tuple(range(2, inp.ndim))).contiguous()
    C = inp.size(0)
    return inp.view(C, -1)


def iou_loss(pred: Tensor, gt: Tensor,
             smooth: Optional[float]=0.01,
             ignore_label: Optional[int]=None) -> Tensor:
    """
    :param pred: after latest activation, the shape is (B, C, spatial_shape)
    :param gt: onehoted gt, the shape is (B, C, spatial_shape)
    :return: IOU, the shape is (B,)
    """
    assert pred.shape == gt.shape
    if ignore_label is not None:
        pred = torch.stack([v for i, v in enumerate(torch.unbind(pred, dim=1))
                            if i != ignore_label])
        gt = torch.stack([v for i, v in enumerate(torch.unbind(gt, dim=1))
                            if i != ignore_label])
    pred = flatten(pred)
    gt = flatten(gt)

    tp = (pred * gt).sum(-1)
    fp = (pred * (1 - gt)).sum(-1)
    fn = ((1 - pred) * gt).sum(-1)

    iou = (tp + smooth) / (tp + fp + fn + EPS + smooth)
    return 1.0 - iou


def generalized_dice_loss(pred: Tensor, gt: Tensor,
                          smooth: Optional[float]=0.01,
                          with_weight: Optional[bool]=True,
                          ignore_label: Optional[int]=None) -> Tensor:
    """
    :param pred: after latest activation, the shape is (B, C, spatial_shape)
    :param gt: onehoted gt, the shape is (B, C, spatial_shape)
    :return: GDice, the shape is (B,)
    """
    assert pred.shape == gt.shape
    if ignore_label is not None:
        pred = torch.stack([v for i, v in enumerate(torch.unbind(pred, dim=1))
                            if i != ignore_label])
        gt = torch.stack([v for i, v in enumerate(torch.unbind(gt, dim=1))
                            if i != ignore_label])

    pred = flatten(pred, with_class=True)
    gt = flatten(gt, with_class=True)

    if with_weight:
        gt_class_flatten = flatten_with_class(gt).sum(-1)
        class_weight = 1.0 / (gt_class_flatten * gt_class_flatten + EPS)
        intersect = (pred * gt).sum(-1) * class_weight.unsqueeze(0)
        intersect = intersect.sum(-1)
    else:
        intersect = (pred * gt).sum([-2, -1])

    # the shape of intersect is (B,)
    # the shape of pred and gt is (B, C, prod(spatial_shape))
    denominator = pred.sum([-2, -1]) + gt.sum([-2, -1])

    assert intersect.shape == denominator.shape, \
        f'{intersect.shape} != {denominator.shape}'
    return 1.0 - (intersect + smooth) / (denominator + EPS + smooth)


def soft_dice_loss(pred: Tensor, gt: Tensor,
                   ignore_label: Optional[int]=None) -> Tensor:
    """
    soft dice = 2 * IOU / (1 + IOU)
    :param pred: after latest activation, the shape is (B, C, spatial_shape)
    :param gt: onehoted gt, the shape is (B, C, spatial_shape)
    :return: dice loss, the shape is (B,)
    """
    iou = iou_loss(pred, gt, ignore_label=ignore_label)
    return 2.0 * iou / (1.0 + iou)


class IOULoss(nn.Module):
    def __init__(self, opt: Opts,
                 activation: Optional[Callable]=None,
                 ignore_label: Optional[int]=None) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        if activation is None:
            self.activation = LogitToPreds(opt)
        else:
            self.activation = activation

    def forward(self, logit: Tensor, gt: Tensor, *,
                reduction: Optional[str]='mean') -> Tensor:
        pred = self.activation(logit)
        loss = iou_loss(pred, gt, ignore_label=self.ignore_label)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f'Unrecognized reduction method ({reduction}).')
        return loss


class GDiceLoss(nn.Module):
    def __init__(self, opt: Opts,
                 activation: Optional[Callable]=None,
                 with_weight: Optional[bool]=False,
                 ignore_label: Optional[int]=None) -> None:
        super().__init__()
        self.with_weight = with_weight
        self.ignore_label = ignore_label
        if activation is None:
            self.activation = LogitToPreds(opt)
        else:
            self.activation = activation
    
    def forward(self, logit: Tensor, gt: Tensor, *,
                onehoted: Optional[bool]=False,
                reduction: Optional[str]='mean') -> Tensor:
        if not onehoted:
            num_classes = logit.size(1)
            with_channel = True if gt.ndim == logit.ndim else False
            onehoted_gt = onehot(gt, num_classes, with_channel=with_channel)
        else:
            onehoted_gt = gt

        pred = self.activation(logit)
        loss = generalized_dice_loss(pred, onehoted_gt,
                                     with_weight=self.with_weight,
                                     ignore_label=self.ignore_label)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f'Unrecognized reduction method ({reduction}).')

        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, opt: Opts,
                 activation: Optional[Callable]=None,
                 ignore_label: int=None,
                 *args, **kwargs) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        if activation is None:
            self.activation = LogitToPreds(opt)
        else:
            self.activation = activation
    
    def forward(self, logit: Tensor, gt: Tensor, *,
                onehoted: Optional[bool]=False,
                reduction: Optional[str]='mean') -> Tensor:
        if not onehoted:
            num_classes = logit.size(1)
            with_channel = True if gt.ndim == logit.ndim else False
            onehoted_gt = onehot(gt, num_classes, with_channel=with_channel)
        else:
            onehoted_gt = gt

        pred = self.activation(logit)
        loss = soft_dice_loss(pred, onehoted_gt,
                              ignore_label=self.ignore_label)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            pass
        else:
            raise ValueError(f'Unrecognized reduction method ({reduction}).')

        return loss


class CrossEntropyLoss(nn.Module):
    def forward(self, logit: Tensor, gt: Tensor, *,
                reduction: Optional[str]='mean',
                ignore_label: Optional[int]=None) -> Tensor:
        assert logit.ndim == gt.ndim + 1

        if ignore_label is None:
            ignore_label = -100
        loss = F.cross_entropy(logit, gt, reduction='none',
                               ignore_index=ignore_label)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(list(range(1, loss.ndim)))
        else:
            raise ValueError(
                f'Unrecognized reduction method ({reduction}).')
        return loss


class BCELossWithLogits(nn.Module):
    def forward(self, logit: Tensor, gt: Tensor, *,
                reduction: Optional[str]='mean',
                ignore_label: Optional[int]=None) -> Tensor:
        assert logit.shape == gt.shape

        if ignore_label is not None:
            logit = torch.stack(
                [v for i, v in enumerate(torch.unbind(logit, dim=1))
                    if i != ignore_label])
            gt = torch.stack(
                [v for i, v in enumerate(torch.unbind(gt, dim=1))
                    if i != ignore_label])

        loss = F.binary_cross_entropy_with_logits(logit, gt, reduction='none')
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(list(range(1, loss.ndim)))
        else:
            raise ValueError(
                f'Unrecognized reduction method ({reduction}).')
        return loss


class GDiceCELoss(nn.Module):
    def __init__(self, opt: Opts, dice_weight: Optional[float]=1.0,
                 ce_weight: Optional[float]=1.0,
                 ignore_label: Optional[int]=None,
                 *args, **kwargs) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_label = ignore_label
        self.dice_loss = GDiceLoss(opt, activation=softmax_helper,
                                   ignore_label=self.ignore_label,
                                   *args, **kwargs)
        self.ce_loss = CrossEntropyLoss()

    def forward(self, logit: Tensor, gt: Tensor, *,
                reduction: Optional[str]='mean') -> Tensor:
        ce_gt = gt.squeeze(1) if logit.ndim == gt.ndim else gt

        dice_loss_ = self.dice_loss(logit, gt.float(), reduction=reduction)
        ce_loss_ = self.ce_loss(logit, ce_gt.long(), reduction=reduction,
                                ignore_label=self.ignore_label)
        loss = dice_loss_ * self.dice_weight + ce_loss_ * self.ce_weight
        return loss


class GDiceBCELoss(nn.Module):
    def __init__(self, opt: Opts, dice_weight: Optional[float]=1.0,
                 ce_weight: Optional[float]=1.0,
                ignore_label: Optional[int]=None,
                *args, **kwargs) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice_loss = GDiceLoss(opt, activation=torch.sigmoid,
                                   ignore_label=self.ignore_label,
                                   *args, **kwargs)
        self.ce_loss = BCELossWithLogits()

    def forward(self, logit: Tensor, gt: Tensor, *,
                reduction: Optional[str]='mean') -> Tensor:
        num_classes = logit.size(1)
        with_channel = True if gt.ndim == logit.ndim else False
        onehoted_gt = onehot(gt, num_classes, with_channel=with_channel)

        dice_loss_ = self.dice_loss(logit, onehoted_gt, onehoted=True,
                                    reduction=reduction)
        ce_loss_ = self.ce_loss(logit, onehoted_gt, reduction=reduction,
                                ignore_label=self.ignore_label)
        loss = dice_loss_ * self.dice_weight + ce_loss_ * self.ce_weight
        return loss


class SoftDiceCELoss(GDiceCELoss):
    def __init__(self, opt: Opts, dice_weight: Optional[float]=1.0,
                 ce_weight: Optional[float]=1.0,
                 ignore_label: Optional[int]=None,
                 *args, **kwargs) -> None:
        super().__init__(opt, dice_weight=dice_weight,
                         ce_weight=ce_weight,
                         ignore_label=ignore_label,
                         *args, **kwargs)
        self.dice_loss = SoftDiceLoss(opt, activation=softmax_helper,
                                      ignore_label=ignore_label,
                                      *args, **kwargs)


class SoftDiceBCELoss(GDiceBCELoss):
    def __init__(self, opt: Opts, dice_weight: Optional[float]=1.0,
                 ce_weight: Optional[float]=1.0,
                 ignore_label: Optional[int]=None,
                 *args, **kwargs) -> None:
        super().__init__(opt, dice_weight=dice_weight,
                         ce_weight=ce_weight,
                         ignore_label=ignore_label,
                         *args, **kwargs)
        self.dice_loss = SoftDiceLoss(opt, activation=torch.sigmoid,
                                      ignore_label=ignore_label, 
                                      *args, **kwargs)


class DeepSupervisedLoss(nn.Module):
    def __init__(self, loss_fn: Callable,
                 weights: Union[List, Tuple]) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, logits: Union[Tuple, List],
                gts: Union[Tuple, List],
                **kwargs) -> Tensor:
        assert len(logits) == len(gts)
        assert len(logits) == len(self.weights)

        final_loss = 0
        for logit, gt, weight in zip(logits, gts, self.weights):
            final_loss += self.loss_fn(logit, gt, **kwargs) * weight

        return final_loss


class LossPicker(object):
    def __init__(self, opt: Opts, *args, **kwargs) -> None:
        super().__init__()
        assert opt.loss in _loss_dict_.keys(), \
            f'{opt.loss} not in {_loss_dict_.keys()}'
        self.loss_fn = _loss_dict_[opt.loss](opt, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.loss_fn(*args, **kwargs)


_loss_dict_ = {
    'IOULoss': IOULoss,
    'GDiceLoss': GDiceLoss,
    'SoftDiceLoss': SoftDiceBCELoss,
    'CrossEntropyLoss': CrossEntropyLoss,
    'BCELossWithLogits': BCELossWithLogits,
    'GDiceCELoss': GDiceCELoss,
    'GDiceBCELoss': GDiceBCELoss,
    'SoftDiceCELoss': SoftDiceCELoss,
    'SoftDiceBCELoss': SoftDiceBCELoss
}
