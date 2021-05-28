from collections import defaultdict
import os
import pickle


class DataContainer(object):
    def __init__(self, name):
        super().__init__()
        self.name = name
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

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._data_dict = data
        return data

    def dump(self, path):
        with open(path, 'wb') as f:
            _data_dict = dict(self._data_dict)
            pickle.dump(_data_dict, f)

    def reset(self):
        self._data_dict = defaultdict(lambda: [])
