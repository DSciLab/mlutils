from collections import defaultdict
import pickle
import numpy as np
import torch


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.data_array = None
        self.cnt = 0
        self.max = -np.Inf
        self.min = np.Inf
        self.avg = None
        self.std = None
        self.latest = None

    def reset(self):
        self.data_array = None
        self.cnt = 0
        self.max = -np.Inf
        self.min = np.Inf
        self.avg = None
        self.std = None
        self.latest = None

    def step(self):
        if self.data_array is not None:
            self.avg = np.mean(self.data_array)
            self.std = np.std(self.data_array)
            if self.avg > self.max:
                self.max = self.avg
            if self.avg < self.min:
                self.min = self.avg
        self.latest = self.avg

    def zero(self):
        self.cnt = 0
        self.data_array = None
        self.avg = 0

    def append(self, data, cnt=1):
        self.cnt += cnt
        if isinstance(data, (np.ndarray, torch.Tensor)):
            data = data.mean()

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if self.data_array is None:
            self.data_array = np.array([data])
        else:
            self.data_array = np.append(self.data_array, data)

    def __str__(self):
        string = f'[{self.name}] '
        if self.data_array is not None:
            string += f'latest: {self.latest:.5f} | '
            string += f'avg: {self.avg:.5f} | '
            string += f'max: {self.max:.5f} | '
            string += f'min: {self.min:.5f} | '
            string += f'std: {self.std:.5f} | '
            string += f'cnt: {self.cnt} |'
        return string


class KFoldMeter(object):
    class Meter(object):
        def __init__(self, name) -> None:
            super().__init__()
            self.name = name
            self.data = None
            self.max = None
            self.min = None
            self.avg = None
            self.std = None
            self.cnt = None

        def append(self, data):
            if self.data is None:
                self.data = np.array([data])
            else:
                self.data = np.append(self.data, data)

        def step(self):
            self.max = np.max(self.data)
            self.min = np.min(self.data)
            self.avg = np.mean(self.data)
            self.std = np.std(self.data)
            self.cnt = len(self.data)

        def __str__(self):
            self.step()
            string = f'[{self.name}] '
            string += f'avg: {self.avg:.5f} | '
            string += f'std: {self.std:.5f} | '
            string += f'min: {self.min:.5f} | '
            string += f'max: {self.max:.5f} | '
            string += f'cnt: {self.cnt:.5f} |'
            return string

    def __init__(self) -> None:
        super().__init__()
        self.fold_meters = defaultdict(lambda: [])
        self._result = None
        self.updated = False

    def reset(self):
        self.fold_meters = defaultdict(lambda: [])
        self._result = None
        self.updated = False

    def append(self, meter_dict):
        assert isinstance(meter_dict, dict)
        for meter_name, meter in meter_dict.items():
            assert isinstance(meter, AverageMeter)
            self.fold_meters[meter_name].append(meter)
        self.updated = True

    @property
    def result(self):
        if self._result is None or self.updated:
            self.updated = False
            self._result = []
            # self.fold_meters = dict(self.fold_meters)
            for meter_name, meter_list in self.fold_meters.items():
                meter = KFoldMeter.Meter(meter_name)
                for curr_fold_meter in meter_list:
                    meter.append(curr_fold_meter.latest)
                self._result.append(meter)
        return self._result

    def __str__(self):
        string = ''
        for meter in self.result:
            string += str(meter) + '\n'
        return string[:-1]

    def dump(self, path):
        data = {'_result': self.result,
                'fold_meters': dict(self.fold_meters)}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.dump(f)
        self.fold_meters = data['fold_meters']
        self._result = data['_result']
