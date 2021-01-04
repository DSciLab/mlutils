from sklearn import metrics
import torch
from torch.nn import functional as F
from mlutils.metrics import *
from sklearn import metrics

class Opts:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes


def test_acc():
    opt = Opts()
    acc_metric = Accuracy(opt)
    pred = torch.rand(100, 5)
    target = torch.randint(0, 5, (100,))

    acc = acc_metric(pred, target)
    acc_true = metrics.accuracy_score(
            target.numpy(), pred.argmax(dim=1).numpy())
    assert acc == acc_true

def test_f1():
    opt = Opts()
    f1_metric = F1Score(opt)
    pred = torch.rand(100, 5)
    target = torch.randint(0, 5, (100,))

    f1 = f1_metric(pred, target)
    f1_true = metrics.f1_score(
            target.numpy(), pred.argmax(dim=1).numpy(),
            average='micro')
    assert f1 == f1_true

def test_au_roc():
    opt = Opts()
    auroc_metric = AUROC(opt)
    pred = torch.randn(100, 5)
    pred = F.softmax(pred, dim=1)
    target = torch.randint(0, 5, (100,))

    auroc = auroc_metric(pred, target)
    auroc_true = metrics.roc_auc_score(
            target.numpy(), pred.numpy(),
            multi_class='ovo')
    assert auroc == auroc_true

def test_kappa():
    opt = Opts()
    kappa_metric = Kappa(opt)
    pred = torch.rand(100, 5)
    target = torch.randint(0, 5, (100,))

    kappa = kappa_metric(pred, target)
    kappa_true = metrics.cohen_kappa_score(
            target.numpy(), pred.argmax(dim=1).numpy())
    assert kappa == kappa_true
