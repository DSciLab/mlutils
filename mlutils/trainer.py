from os import terminal_size
from tqdm import trange


class Trainer(object):
    def __init__(self, opt, model, optimizer, scheduler=None):
        self.opt = opt
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.step = 0

    def train(self, train_loader, eval_loader=None):
        initial_epoch = self.epoch
        for _ in range(initial_epoch, self.opt.epochs):
            self.epoch += 1

    def eval(self, eval_loader):
        pass

    def train_epoch(self, data_loader):
        self.model.train()
        data_len = len(data_loader)
        with trange(data_len) as t:
            for item in data_loader:
                self.step += 1
                t.set_description(
                    f'Training {self.step} | '
                    f'[{self.epoch}/{self.opt.epochs}]'
                )
                pass

    def eval_epoch(self, data_loader):
        self.model.eval()
        data_len = len(data_loader)
        with trange(data_len) as t:
            for item in data_loader:
                self.step += 1
                t.set_description(
                    f'Validation {self.step} | '
                    f'[{self.epoch}/{self.opt.epochs}]'
                )
                pass

    def train_step(self, item):
        raise NotImplementedError

    def eval_step(self, item):
        raise NotImplementedError

    def inference(self, image):
        raise NotImplementedError

    def on_epoch_begin(self):
        raise NotImplementedError

    def on_epoch_end(self):
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
