import time

from numpy.lib.stride_tricks import DummyArray


class StopWatch(object):
    def __init__(self):
        self._start = 0
        self._history = []

    def start(self):
        self._start = time.perf_counter()

    def reset(self):
        self._start = time.perf_counter()

    def lap(self):
        curr = time.perf_counter()
        duration = curr - self._start
        self._history.append(duration)
        return duration

    def perfect_lap(self):
        duration = self.lap()
        H = 60*60
        M = 60
        
        h = int(duration / H)
        rm = duration % H
        m = int(rm / M)
        s = rm % M

        return f'{h}:{m}:{s:.2f}'
