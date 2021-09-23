import threading as T
import datetime
from typing import List, Union
import torch
import numpy as np
import inspect
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from mlutils.dataloader import DataLoader
from tqdm import trange
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn
from .stop_watch import StopWatch
from .log import Log
from .metrics import Metric
from .meter import AverageMeter
from .dashboard import Dashobard
from .saver import Saver
from .container import DataContainer
from .utils import LogitToPreds
from . import gen


def detach_cpu(fn):
    def __fn(*args, **kwargs):
        outputs = fn(*args, **kwargs)
        if isinstance(outputs, tuple):
            return_list = []
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    return_list.append(output.detach().cpu())
                else:
                    return_list.append(output)
            return tuple(return_list)
        else:
            if isinstance(outputs, torch.Tensor):
                return outputs.detach().cpu()
            else:
                return outputs
    return __fn


class Trainer(object):
    REPORT_LOCK = T.Lock()
    TRACE_FREQ = 41

    def __init__(self, opt, device_id=None):
        self.device_id = device_id
        self.opt = opt
        self.debug = opt.get('debug', False)
        self.enable_progressbar = True
        self.saver = Saver(opt)
        self.stop_watch = StopWatch()
        self.dashboard = Dashobard(opt)
        self.epochs = opt.epochs
        self.epoch = 0
        self.step = 0
        self.metrics = []
        self.train_meters = {}
        self.eval_meters = {}
        self.test_meters = {}
        self.nn_models = {}
        self.nn_optimizers = {}
        self.lr_schedulers = {}
        self.training = True
        self.testing = False
        self.best = False
        self.train_loader = None
        self.eval_loader = None
        self.latest_loss = np.Inf
        self.min_loss = np.Inf
        self.curr_fold = 0
        self.eval_no_grad = True
        self.eval_container = DataContainer('eval_data')
    
        train_loss_meter = AverageMeter('train_loss')
        eval_loss_meter = AverageMeter('eval_loss')
        test_loss_meter = AverageMeter('test_loss')
        self.train_meters['loss'] = train_loss_meter
        self.eval_meters['loss'] = eval_loss_meter
        self.test_meters['loss'] = test_loss_meter
        self.logit_to_preds = LogitToPreds(opt)

        self.saver.save_cfg(opt)

        if self.opt.get('device', None) is None:
            self.device = torch.device(f'cpu')
        else:
            self.device = torch.device(f'cuda:0')

        # Log.info('Initiated Trainer')
        # Log.info(f'ID: {opt.id}')
        # Log.debug(opt.perfect())
        try:
            self.setup()
        except NotImplementedError:
            pass

    def setup(self):
        raise NotImplementedError

    @gen.asynchrony
    def to_gpu(self, obj, parallel=False, gpu_id=None):
        gpu_id = gpu_id or self.device_id
        if self.opt.get('device', None) is None:
            return obj
        else:
            num_gpu = torch.cuda.device_count()
            # print('num_gpu', num_gpu)
            if parallel and num_gpu > 1:
                # for network
                obj = obj.cuda()
                obj = torch.nn.DataParallel(
                    obj, device_ids=list(range(num_gpu)))
                return obj
            else:
                # for data
                if isinstance(obj, (list, tuple)):
                    objs = []
                    for _obj in obj:
                        objs.append(self._data_to_gpu(gpu_id, _obj))
                    return objs
                else:
                    return self._data_to_gpu(gpu_id, obj)

    @staticmethod
    def _data_to_gpu(gpu_id, obj):
        if gpu_id is None:
            return obj.cuda()
        else:
            device = torch.device(f'cuda:{gpu_id}')
            return obj.to(device)

    @gen.asynchrony
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

        if isinstance(value, AverageMeter):
            if not hasattr(self, 'avg_meters'):
                super().__setattr__('avg_meters', {})
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
            test_meter = AverageMeter(metric.__class__.__name__)
            self.metrics.append(metric)
            self.train_meters[train_meter.name] = train_meter
            self.eval_meters[eval_meter.name] = eval_meter
            self.test_meters[test_meter.name] = test_meter

    @gen.asynchrony
    @gen.synchrony
    def metric(self, preds, labels):
        if self.training:
            meters = self.train_meters
        elif self.testing:
            meters = self.test_meters
        else:
            meters = self.eval_meters

        preds = yield preds
        labels = yield labels

        if preds is not None and labels is not None:
            for metric in self.metrics:
                meter = meters[metric.__class__.__name__]
                meter.append(metric(preds, labels))

    def _report_epoch(self):
        self.REPORT_LOCK.acquire()
        if self.opt.get('train_mod', 'split') == 'k_fold':
            Log.info('==================')
            Log.info(f'   Fold: {self.curr_fold}')
            Log.info(f'   Epoch: {self.epoch}')
            Log.info('==================')
        try:
            self.report_epoch()
        except NotImplementedError:
            pass

        Log.info(f'ID: {self.opt.id}')
        Log.info(f'LR: {self.current_lr_str}')
        Log.info(f'Epoch: {self.epoch}/{self.epochs}')
        Log.info(f'Now: {datetime.datetime.now().ctime()}')
        Log.info(f'Duration: {self.stop_watch.perfect_lap()}')
        # if self.dashboard.enabled:
        #     Log.info(f'Dashboard: {self.dashboard.address}')
        for meter in self.eval_meters.values():
            Log.info(meter)
        self.REPORT_LOCK.release()

    def report_epoch(self):
        raise NotImplementedError

    def set_fold(self, k):
        self.curr_fold = k
        self.saver.set_fold(k)

    def update_transformer_param(self):
        if self.train_loader is not None and \
            isinstance(self.train_loader, DataLoader):
            self.train_loader.update_transformer(
                verbose=self.opt.get('debug', False))

        if self.train_loader is not None and \
            isinstance(self.eval_loader, DataLoader):
            self.eval_loader.update_transformer(
                verbose=self.opt.get('debug', False))

    def eval_state(self):
        self.training = False
        self.dashboard.eval()
        for model in self.nn_models.values():
            model.eval()
            model.zero_grad()
        for meter in self.eval_meters.values():
            meter.zero()

    def training_state(self):
        self.training = True
        self.dashboard.train()
        for model in self.nn_models.values():
            model.train()
        for meter in self.train_meters.values():
            meter.zero()

    def train(self, train_loader, eval_loader=None):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        if train_loader is not None:
            Log.info(f'Training dataloader length: {len(train_loader)}')
        if eval_loader is not None:
            Log.info(f'Eval dataloader length: {len(eval_loader)}')

        try:
            self.on_training_begin()
        except NotImplementedError:
            pass

        initial_epoch = self.epoch
        self.stop_watch.start()
        for epoch in range(initial_epoch, self.epochs):
            self.epoch = epoch
            try:
                self.on_epoch_begin()
            except NotImplementedError:
                pass
            self.dashboard.step()

            self.train_epoch(self.train_loader)
            if self.eval_loader is not None:
                if self.eval_no_grad:
                    with torch.no_grad():
                        self.eval_epoch(self.eval_loader)
                else:
                    self.eval_epoch(self.eval_loader)
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

    def eval(self, eval_loader):
        with torch.no_grad():
            self.eval_epoch(eval_loader)

    @gen.synchrony
    def train_epoch(self, data_loader):
        self.training_state()

        metric_futures = gen.FutureList()
        if isinstance(data_loader, (DataLoader, TorchDataLoader)):
            data_len = len(data_loader)
            with trange(data_len, disable=(not self.enable_progressbar)) as t:
                for item in data_loader:
                    self.step += 1
                    rets = self.train_step(item)
                    loss, preds, labels = rets[:3]
                    metric_future = self.metric(preds, labels)
                    metric_futures.append(metric_future)
                    loss = yield loss
                    self.train_meters['loss'].append(loss)
                    t.set_description(
                        f'Training '
                        f'[{self.step}/{self.epoch}/{self.opt.epochs}] '
                        f'[loss: {loss:.3f}]'
                        f'[lr: {self.current_lr_str}]'
                    )
                    t.update()
        else:
            if hasattr(self, 'num_batch_per_epoch_train'):
                num_batch_per_epoch = self.num_batch_per_epoch_train
            else:
                num_batch_per_epoch = 250

            with trange(num_batch_per_epoch, disable=(not self.enable_progressbar)) as t:
                for _ in t:
                    item = next(data_loader)
                    self.step += 1
                    rets = self.train_step(item)
                    loss, preds, labels = rets[:3]
                    metric_future = self.metric(preds, labels)
                    metric_futures.append(metric_future)
                    loss = yield loss
                    self.train_meters['loss'].append(loss)
                    t.set_description(
                        f'Training '
                        f'[{self.step}/{self.epoch}/{self.opt.epochs}] '
                        f'[loss: {loss:.3f}]'
                        f'[lr: {self.current_lr_str}]'
                    )

        yield metric_futures
        for meter in self.train_meters.values():
            meter.step()
            self.dashboard.add_meter(meter, fold=self.curr_fold,
                                     training=True)

    @gen.asynchrony
    @gen.synchrony
    def eval_container_append(self, preds, labels):
        preds = yield preds
        labels = yield labels
        if preds is not None and labels is not None and not self.debug:
            self.eval_container.append({'preds': preds.numpy(),
                                        'labels': labels.numpy()})

    @gen.synchrony
    def eval_epoch(self, data_loader):
        self.eval_state()

        futures_list = gen.FutureList()
        if isinstance(data_loader, (DataLoader, TorchDataLoader)):
            data_len = len(data_loader)
            with trange(data_len, disable=(not self.enable_progressbar)) as t:
                for item in data_loader:
                    rets = self.eval_step(item)
                    loss, preds, labels = rets[:3]
                    loss = yield loss
                    future_container = self.eval_container_append(preds, labels)
                    futures_list.append(future_container)
                    future_metric = self.metric(preds, labels)
                    futures_list.append(future_metric)
                    self.eval_meters['loss'].append(loss)
                    t.set_description(
                        f'Validation [{self.step}/'
                        f'{self.epoch}/{self.opt.epochs}]'
                    )
                    t.update()
                    # break # For debugging
        else:
            if hasattr(self, 'num_batch_per_epoch_val'):
                num_batch_per_epoch = self.num_batch_per_epoch_val
            else:
                num_batch_per_epoch = 50
            with trange(num_batch_per_epoch, disable=(not self.enable_progressbar)) as t:
                for _ in t:
                    item = next(data_loader)
                    rets = self.eval_step(item)
                    loss, preds, labels = rets[:3]
                    loss = yield loss
                    future_container = self.eval_container_append(preds, labels)
                    futures_list.append(future_container)
                    future_metric = self.metric(preds, labels)
                    futures_list.append(future_metric)
                    self.eval_meters['loss'].append(loss)
                    t.set_description(
                        f'Validation [{self.step}/'
                        f'{self.epoch}/{self.opt.epochs}]'
                    )
                    # break # For debugging

        yield futures_list
        for meter in self.eval_meters.values():
            meter.step()
            self.dashboard.add_meter(meter, fold=self.curr_fold, training=False)

        loss_meter = self.eval_meters['loss']
        self.latest_loss = loss_meter.avg
        if self.latest_loss < self.min_loss:
            self.best = True
            self.min_loss = self.latest_loss
        else:
            self.best = False

    def test(self, test_dataloader):
        try:
            self.on_testing_begin()
        except NotImplementedError:
            pass
        self.training = False
        self.testing = True
        self.eval_state()
        test_container = DataContainer('testResult')
        test_stop_watch = StopWatch()

        data_len = len(test_dataloader)
        cnt = 0
        with trange(data_len) as t:
            for item in test_dataloader:
                assert isinstance(item, (tuple, list)) and len(item) >= 2, \
                    f'Testing dataset return should be a instance of tuple and length of ' + \
                    f'the tuple should be >= 2, but the type of item is {type(item)} and ' + \
                    f'length of item is {len(item)}.'
                case_id = item[0][0]
                item = item[1:]
                assert isinstance(case_id, str), \
                    f'The type of case_id should be str, but {type(case_id)} got.'
                with torch.no_grad():
                    test_stop_watch.start()
                    rets = self.inference(item)
                    test_stop_watch.lap()
                    preds, labels = rets
                    data = item[0]
                    test_container.append({'preds': preds.cpu().numpy(),
                                           'labels': labels.cpu().numpy(),
                                           'data': data.numpy(),
                                           'case_id': case_id})
                    self.metric(preds, labels)
                t.set_description(
                    f'[Testing {cnt}/{data_len}]'
                )
                t.update()
                cnt += 1

        for meter in self.test_meters.values():
            meter.step()
            Log.info(meter)

        Log.info(test_stop_watch.show_statistic())
        self.saver.save_container(test_container)
        self.testing = False
        try:
            self.on_testing_end()
        except NotImplementedError:
            pass

    def inference(self, vox):
        raise NotImplementedError

    def _current_lr_str(self, lr) -> str:
        lr = self.current_lr
        if isinstance(lr, float):
            return f'{lr:.7f}'
        elif isinstance(lr, list):
            s = ''
            for _lr in lr:
                s += f'{_lr:.7f} | '
            return s[:-2]
        elif isinstance(lr, dict):
            s = ''
            for k, v in lr.items():
                s += f'{k}: {self._current_lr_str(v)} | '
            return s[:-2]
        else:
            raise ValueError(f'Invalid lr type ({type(lr)})')

    @property
    def current_lr_str(self) -> str:
        lr = self.current_lr
        return self._current_lr_str(lr)

    @property
    def current_lr(self) -> Union[float, List[float]]:
        result = {}
        for key, val in self.nn_optimizers.items():
            lrs = [para['lr'] for para in val.param_groups]

            if len(lrs) == 1:
                lr = lrs[0]
            else:
                lr = lrs
            result[key] = lr

        if len(result) == 1:
            return list(result.values())[0]
        else:
            return result

    curr_lr = current_lr
    lr = current_lr

    def save_stat_dict(self):
        if not self.debug:
            state_dict = self.state_dict()
            self.saver.save_state_dict(state_dict, best=self.best)
            self.saver.save_container(self.eval_container, best=self.best)
            self.eval_container.reset()

    def train_step(self, item):
        # return loss, preds, labels, ....
        raise NotImplementedError

    def eval_step(self, item):
        # return loss, preds, labels, ....
        raise NotImplementedError

    def inference(self, image):
        raise NotImplementedError

    def on_testing_end(self):
        raise NotImplementedError

    def on_testing_end(self):
        raise NotImplementedError

    def on_epoch_begin(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError

    def on_training_begin(self):
        raise NotImplementedError

    def on_training_end(self):
        raise NotImplementedError

    def on_testing_begin(self):
        raise NotImplementedError

    def on_testing_end(self):
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
            Log.info(f'Loading epoch.')
            self.epoch = state_dict['epoch']
        else:
            Log.info(f'Failed to load epoch.')

        # load model
        for key, model in self.nn_models.items():
            if self._check_state_dict(state_dict, key, strict):
                Log.info(f'Loading model {key}')
                model.load_state_dict(state_dict[key])
            else:
                Log.info(f'Failed to load model {key}')

        # load optimizer
        for key, optimizer in self.nn_optimizers.items():
            if self._check_state_dict(state_dict, key, strict):
                Log.info(f'Loading optimizer {key}')
                optimizer.load_state_dict(state_dict[key])
            else:
                Log.info(f'Failed to load optimizer {key}')

        # load scheduler
        for key, scheduler in self.lr_schedulers.items():
            if self._check_state_dict(state_dict, key, strict):
                Log.info(f'Loading scheduler {key}')
                scheduler.load_state_dict(state_dict[key])
            else:
                Log.info(f'Failed to load scheduler {key}')

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
