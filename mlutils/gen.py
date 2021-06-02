import torch
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Generator
from .log import Log


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ProcessPool(object, metaclass=Singleton):
    MAX_WORKERS = 10
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)

    def submit(self, func, *args, **kwargs):
        return self.executor.submit(func, *args, **kwargs)


class FutureList(object):
    def __init__(self) -> None:
        super().__init__()
        self.future_list = []

    def append(self, future):
        assert isinstance(future, Future), \
            'Typer error, future is not an instance of Future, ' + \
            f'type(future)={type(future)}'
        self.future_list.append(future)

    def consume_all(self):
        result_list = []
        for future in self.future_list:
            res = future.result()
            result_list.append(res)
        return result_list

    result = consume_all


def set_max_workers(max_workers):
    ProcessPool.MAX_WORKERS = max_workers


def synchrony(func):
    def _exec(*args, **kwargs):
        try:
            gen = func(*args, **kwargs)
        except Exception as e:
            Log.error(func)
            raise e
        if isinstance(gen, Generator):
            res = next(gen)
            while True:
                try:
                    if isinstance(res, (Future, FutureList)):
                        res = gen.send(res.result())
                    else:
                        res = gen.send(res)
                except StopIteration as e:
                    return e.value
        else:
            return gen
    return _exec


def asynchrony(func):
    def _exec(*args, **kwargs):
        return ProcessPool().submit(func, *args, **kwargs)
    return _exec


def _detach_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    else:
        return obj


def detach_cpu(func):
    def _exec(*args, **kwargs):
        outputs = func(*args, **kwargs)
        # if isinstance(outputs, StopIteration):
        #     outputs = outputs.value

        if isinstance(outputs, (tuple, list)):
            return_list = []
            for output in outputs:
                future = ProcessPool().submit(_detach_cpu, output)
                return_list.append(future)
            return tuple(return_list)
        else:
            return ProcessPool().submit(_detach_cpu, outputs)
    return _exec
