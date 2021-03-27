from collections import defaultdict
import os
import pickle


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
        self._data_dict = defaultdict(lambda: [])

    def __getitem__(self, key):
        return self._data_dict[key]

    def __setitem__(self, key, value):
        self._data_dict[key] = value

    def keys(self):
        return self._data_dict.keys()

    def items(self, *keys):
        zip_items = [self[key] for key in keys]
        for item in zip(*zip_items):
            yield item

    def append(self, data_dict):
        assert isinstance(data_dict, dict)
        for key in data_dict.keys():
            self._data_dict[key].append(data_dict[key])

    def load(self):
        with open(self.target_path, 'rb') as f:
            data = pickle.load(f)
        self._data_dict = data
        return data

    def dump(self):
        with open(self.target_path, 'wb') as f:
            _data_dict = dict(self._data_dict)
            pickle.dump(_data_dict, f)

    def reset(self):
        self._data_dict = defaultdict(lambda: [])
