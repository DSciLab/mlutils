import os
import numpy as np
from mlutils.container import DataContainer


class Opts(object):
    def __init__(self) -> None:
        super().__init__()
        self.saver_root = '/tmp'
        self.id = 'test_id'
    
    def get(self, key, default=None):
        return getattr(self, key)


opt = Opts()


def test_append_data():
    container = DataContainer('test', opt)

    for _ in range(5):
        pred = np.random.rand(20)
        target = np.random.rand(20)
        container.append({'pred': pred,
                          'target': target})
    
    assert len(container['pred']) == 20 * 5
    assert len(container['target']) == 20 * 5


def test_dump_load_data():
    container = DataContainer('test', opt)

    for _ in range(5):
        pred = np.random.rand(20)
        target = np.random.rand(20)
        container.append({'pred': pred,
                          'target': target})
    
    container.dump()
    new_container = DataContainer('test', opt)    
    new_container.load()

    assert len(new_container['pred']) == 20 * 5
    assert len(new_container['target']) == 20 * 5

    os.remove(new_container.target_path)