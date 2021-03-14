import os
import pickle
import numpy as np


class DataContainer(object):
    DEFAULT_ROOT = '.saver'

    def __init__(self, name, opt):
        super().__init__()
        root = opt.get('saver_root', self.DEFAULT_ROOT)
        target_dir = os.path.join(root, opt.id)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        self.name = name
        self.target_path = os.path.join(target_dir, f'{name}_latest.pickle')
        self._data_dict = {}

    def __getitem__(self, key):
        return self._data_dict[key]

    def __setitem__(self, key, value):
        self._data_dict[key] = value

    def append(self, data_dict):
        assert isinstance(data_dict, dict)

        if not self._data_dict:
            self._data_dict = data_dict
        else:
            assert sorted(self._data_dict.keys()) \
                == sorted(data_dict.keys())
            for key in data_dict.keys():
                self._data_dict[key] = np.concatenate([self._data_dict[key],
                                                       data_dict[key]],
                                                      axis=0)

    def load(self):
        with open(self.target_path, 'rb') as f:
            data = pickle.load(f)
        self._data_dict = data
        return data

    def dump(self):
        with open(self.target_path, 'wb') as f:
            pickle.dump(self._data_dict, f)
