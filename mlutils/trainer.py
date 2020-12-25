from tqdm import trange
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
from .stop_watch import StopWatch
from .log import Log
from .metrics import *
from .meter import AverageMeter


def entry(fn):
    def __fn(*args, **kwargs):
        obj = args[0]
        obj.init()
        fn(*args, **kwargs)
    return __fn


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.stop_watch = StopWatch()
        self.epoch = 0
        self.step = 0
        self.metrics = []
        self.train_meters = {}
        self.eval_meters = {}
        self.nn_models = {}
        self.nn_optimizers = {}
        self.lr_schedulers = {}
        self.training = True
    
    def init(self):
        train_loss_meter = AverageMeter('train_loss')
        eval_loss_meter = AverageMeter('eval_loss')
        self.train_meters['loss'] = train_loss_meter
        self.train_meters['eval'] = eval_loss_meter

    def __setattr__(self, name, value):
        # collect nn models
        if isinstance(value, nn.Module):
            if not hasattr(self, 'nn_models'):
                super().__setattr__('nn_models', {})
            self.nn_models[name] = value

        # collect optimzier
        if isinstance(value, Optimizer):
            if not hasattr(self, 'nn_optimizers'):
                super().__setattr__('nn_optimizers', {})
            self.nn_optimizers[name] = value

        # collect lr scheduler
        if isinstance(value, _LRScheduler):
            if not hasattr(self, 'lr_schedulers'):
                super().__setattr__('lr_schedulers', {})
            self.lr_schedulers[name] = value

        super().__setattr__(name, value)

    def set_metrics(self, *metric_classes):
        for metric_cls in metric_classes:
            metric = metric_cls(self.opt)
            train_meter = AverageMeter(metric_cls.__name__)
            eval_meter = AverageMeter(metric_cls.__name__)
            self.metrics.append(metric)
            self.train_meters[train_meter.name] = train_meter
            self.eval_meters[eval_meter.name] = eval_meter

    def metrics(self, preds, labels):
        if self.training:
            meters = self.train_meters
        else:
            meters = self.eval_meters

        for metric in self.metrics:
            meter = meters[metric.__class__.__name__]
            meter.append(metric(preds, labels))

    def report_epoch(self):
        Log.info(f'Duration: {self.stop_watch.prefect_lap()}')
        for meter in self.eval_meters:
            Log.info(meter)

    @entry
    def train(self, train_loader, eval_loader=None):
        initial_epoch = self.epoch
        self.stop_watch.start()
        for _ in range(initial_epoch, self.opt.epochs):
            self.epoch += 1
            self.on_epoch_begin()
            self.train_epoch(train_loader)
            if eval_loader is not None:
                self.eval_epoch(eval_loader)
            self.report_epoch()
            self.on_epoch_end()

    @entry
    def eval(self, eval_loader):
        self.eval_epoch(eval_loader)

    def train_epoch(self, data_loader):
        self.training = True
        self.model.train()
        data_len = len(data_loader)
        with trange(data_len) as t:
            for item in data_loader:
                self.step += 1
                rets = self.train_step(item)
                loss, preds, labels = rets[:3]
                self.metrics(preds, labels)
                self.train_meters['loss'].append(loss)
                t.set_description(
                    f'Training '
                    f'[{self.step}/{self.epoch}/{self.opt.epochs}] '
                    f'[loss: [{loss:.3f}]]'
                )

    def eval_epoch(self, data_loader):
        self.training = False
        self.model.eval()
        data_len = len(data_loader)
        with trange(data_len) as t:
            for item in data_loader:
                rets = self.train_step(item)
                loss, preds, labels = rets[:3]
                self.metrics(preds, labels)
                self.eval_meters['loss'].append(loss)
                t.set_description(
                    f'Validation {self.step} | '
                    f'[{self.epoch}/{self.opt.epochs}]'
                )

    def train_step(self, item):
        # return loss, preds, labels, ....
        raise NotImplementedError

    def eval_step(self, item):
        # return loss, preds, labels, ....
        raise NotImplementedError

    def inference(self, image):
        raise NotImplementedError

    def on_epoch_begin(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError

    def on_training_begin(self):
        raise NotImplementedError

    def on_training_end(self):
        raise NotImplementedError

    def _check_state_dict(self, state_dict, key, strict):
        if strict:
            if not ((hasattr(self, key) and key in state_dict.keys()) or \
            (not hasattr(self, key) and key not in state_dict.keys())):
                if hasattr(self, key):
                    raise RuntimeError(f'Expecte {key} in state_dict.')
                else:
                    raise RuntimeError(f'Unexpected {key} in state_dict.')
        if hasattr(self, key) and key in state_dict.keys():
            return True
        else:
            return False

    def load_state_dict(self, state_dict, strict=True):
        if self._check_state_dict(state_dict, 'epoch', strict):
            self.epoch = state_dict['epoch']

        if self._check_state_dict(state_dict, 'model', strict):
            self.model.load_state_dict(state_dict['model'])

        if self._check_state_dict(state_dict, 'optimzier', strict):
            self.optimzier.load_state_dict(state_dict['optimzier'])
        
        if self._check_state_dict(state_dict, 'scheduler', strict):
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def state_dict(self):
        state = {}
        if hasattr(self, 'epoch'):
            state['epoch'] = self.epoch
        if hasattr(self, 'model'):
            state['model'] = self.model.state_dict()
        if hasattr(self, 'optimizer'):
            state['optimizer'] = self.optimizer.state_dict()
        if hasattr(self, 'scheduler'):
            state['scheduler'] = self.scheduler.state_dict()

        return state
