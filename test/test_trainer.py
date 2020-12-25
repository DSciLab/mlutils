from torch import optim
from torch.optim import lr_scheduler
import torch
from torch import nn
from mlutils.trainer import Trainer


class Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 3)


def test_models_collect():
    _mods = {}

    class TestTrainer(Trainer):
        def __init__(self):
            super().__init__(None)
            self.mod1 = Mod()
            self.mod2 = Mod()
            _mods['mod1'] = self.mod1
            _mods['mod2'] = self.mod2

    trainer = TestTrainer()
    assert len(trainer.nn_models) == 2

    assert trainer.nn_models['mod1'] == _mods['mod1']
    assert trainer.nn_models['mod2'] == _mods['mod2']


def test_optimizers_collect():
    _optims = {}

    class TestTrainer(Trainer):
        def __init__(self):
            super().__init__(None)
            self.optim1 = optim.SGD(Mod().parameters(), 0.1)
            self.optim2 = optim.Adam(Mod().parameters(), 0.0001)
            _optims['optim1'] = self.optim1
            _optims['optim2'] = self.optim2

    trainer = TestTrainer()
    assert len(trainer.nn_models) == 0
    assert len(trainer.nn_optimizers) == 2

    assert trainer.nn_optimizers['optim1'] == _optims['optim1']
    assert trainer.nn_optimizers['optim2'] == _optims['optim2']


def test_schedulers_collect():
    _schs = {}
    _optims = {}

    class TestTrainer(Trainer):
        def __init__(self):
            super().__init__(None)
            self.optim1 = optim.SGD(Mod().parameters(), 0.1)
            self.optim2 = optim.Adam(Mod().parameters(), 0.0001)
            _optims['optim1'] = self.optim1
            _optims['optim2'] = self.optim2
            _schs['sch1'] = lr_scheduler.CyclicLR(self.optim1, 0.1, 0.2)
            _schs['sch2'] = lr_scheduler.ExponentialLR(self.optim2, gamma=0.5)
            self.sch1 = _schs['sch1']
            self.sch2 = _schs['sch2']

    trainer = TestTrainer()
    assert len(trainer.nn_models) == 0
    assert len(trainer.nn_optimizers) == 2
    assert len(trainer.lr_schedulers) == 2

    assert trainer.lr_schedulers['sch1'] == _schs['sch1']
    assert trainer.lr_schedulers['sch2'] == _schs['sch2']

    assert trainer.nn_optimizers['optim1'] == _optims['optim1']
    assert trainer.nn_optimizers['optim2'] == _optims['optim2']
