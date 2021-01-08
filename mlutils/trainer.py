import numpy as np
import inspect
from tqdm import trange
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
from .stop_watch import StopWatch
from .log import Log
from .metrics import ECE, Metric
from .meter import AverageMeter
from .dashboard import Dashobard
from .saver import Saver


def entry(fn):
    def __fn(*args, **kwargs):
        fn(*args, **kwargs)
    return __fn


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.stop_watch = StopWatch()
        self.dashboard = Dashobard(opt)
        self.epoch = 0
        self.step = 0
        self.saver = Saver(opt)
        self.metrics = []
        self.train_meters = {}
        self.eval_meters = {}
        self.nn_models = {}
        self.nn_optimizers = {}
        self.lr_schedulers = {}
        self.training = True
        self.best = False
        self.latest_loss = np.Inf
    
        train_loss_meter = AverageMeter('train_loss')
        eval_loss_meter = AverageMeter('eval_loss')
        self.train_meters['loss'] = train_loss_meter
        self.eval_meters['loss'] = eval_loss_meter

        self.saver.save_cfg(opt)

    def to_gpu(self, obj):
        if self.opt.get('device', None) is None:
            return obj
        else:
            return obj.cuda()

    def to_cpu(self, obj):
        return obj.cpu()

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
            if inspect.isclass(metric_cls) and issubclass(metric_cls, Metric):
                metric = metric_cls(self.opt)
            elif isinstance(metric_cls, Metric):
                metric = metric_cls
            else:
                raise RuntimeError(f'Unrecognized metric')
            
            train_meter = AverageMeter(metric.__class__.__name__)
            eval_meter = AverageMeter(metric.__class__.__name__)
            self.metrics.append(metric)
            self.train_meters[train_meter.name] = train_meter
            self.eval_meters[eval_meter.name] = eval_meter

    def metric(self, preds, labels):
        if self.training:
            meters = self.train_meters
        else:
            meters = self.eval_meters

        for metric in self.metrics:
            meter = meters[metric.__class__.__name__]
            meter.append(metric(preds, labels))

    def _report_epoch(self):
        try:
            self.report_epoch()
        except NotImplementedError:
            pass

        Log.info(f'ID: {self.opt.id}')
        Log.info(f'Duration: {self.stop_watch.perfect_lap()}')
        if self.dashboard.enabled:
            Log.info(f'Dashboard: {self.dashboard.address}')
        for meter in self.eval_meters.values():
            Log.info(meter)

    def report_epoch(self):
        raise NotImplementedError

    @entry
    def train(self, train_loader, eval_loader=None):
        try:
            self.on_training_begin()
        except NotImplementedError:
            pass

        initial_epoch = self.epoch
        self.stop_watch.start()
        for _ in range(initial_epoch, self.opt.epochs):
            self.epoch += 1
            try:
                self.on_epoch_begin()
            except NotImplementedError:
                pass
            self.dashboard.step()
            self.train_epoch(train_loader)
            if eval_loader is not None:
                self.eval_epoch(eval_loader)
            self.save_stat_dict()
            self._report_epoch()
            try:
                self.on_epoch_end()
            except NotImplementedError:
                pass
        try:
            self.on_training_end()
        except NotImplementedError:
            pass

    @entry
    def eval(self, eval_loader):
        self.eval_epoch(eval_loader)

    def train_epoch(self, data_loader):
        self.training = True
        self.dashboard.train()
        for model in self.nn_models.values():
            model.train()

        for meter in self.train_meters.values():
            meter.zero()

        data_len = len(data_loader)
        with trange(data_len) as t:
            for item in data_loader:
                self.step += 1
                rets = self.train_step(item)
                loss, preds, labels = rets[:3]
                self.metric(preds, labels)
                self.train_meters['loss'].append(loss)
                t.set_description(
                    f'Training '
                    f'[{self.step}/{self.epoch}/{self.opt.epochs}] '
                    f'[loss: {loss:.3f}]'
                )
                t.update()

        for meter in self.train_meters.values():
            meter.step()
            self.dashboard.add_meter(meter)

    def eval_epoch(self, data_loader):
        self.training = False
        self.dashboard.eval()
        for model in self.nn_models.values():
            model.eval()
        for meter in self.eval_meters.values():
            meter.zero()

        data_len = len(data_loader)
        with trange(data_len) as t:
            for item in data_loader:
                rets = self.eval_step(item)
                loss, preds, labels = rets[:3]
                self.metric(preds, labels)
                self.eval_meters['loss'].append(loss)
                t.set_description(
                    f'Validation {self.step} | '
                    f'[{self.epoch}/{self.opt.epochs}]'
                )
                t.update()

        for meter in self.eval_meters.values():
            meter.step()
            self.dashboard.add_meter(meter)

        loss_meter = self.eval_meters['loss']
        latest_loss = loss_meter.avg
        if latest_loss < self.latest_loss:
            self.best = True
        else:
            self.best = False
        self.latest_loss = latest_loss

    def save_stat_dict(self):
        state_dict = self.state_dict()
        self.saver.save_state_dict(state_dict, best=self.best)

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

        # load model
        for key, model in self.nn_models.items():
            if self._check_state_dict(state_dict, key, strict):
                model.load_state_dict(state_dict[key])

        # load optimizer
        for key, optimizer in self.nn_optimizers.items():
            if self._check_state_dict(state_dict, key, strict):
                optimizer.load_state_dict(state_dict[key])
        
        # load scheduler
        for key, scheduler in self.lr_schedulers.items():
            if self._check_state_dict(state_dict, key, strict):
                scheduler.load_state_dict(state_dict[key])

    def state_dict(self):
        state = {}
        if hasattr(self, 'epoch'):
            state['epoch'] = self.epoch

        for key, model in self.nn_models.items():
            state[key] = model.state_dict()
        for key, optimizer in self.nn_optimizers.items():
            state[key] = optimizer.state_dict()
        for key, scheduler in self.lr_schedulers.items():
            state[key] = scheduler.state_dict()

        return state

    def load_state(self):
        model = self.opt.model
        state_dict = self.saver.load_state_dict(model)
        self.load_state_dict(state_dict)
