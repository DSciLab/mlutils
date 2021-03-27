import time
import numpy as np


class StopWatch(object):
    def __init__(self):
        self._start = 0
        self._latest = 0
        self._duration_from_start_history = []
        self._duration_from_latest_history = []

    def start(self):
        self._start = time.perf_counter()
        self._latest = self._start

    def reset(self):
        self._start = time.perf_counter()
        self._latest = self._start

    def lap(self):
        curr = time.perf_counter()
        duration_from_start = curr - self._start
        duration_from_latest = curr - self._latest
        self._duration_from_start_history.append(duration_from_start)
        self._duration_from_latest_history.append(duration_from_latest)
        self._latest = curr
        return duration_from_start

    def perfect_lap(self):
        duration = self.lap()
        H = 60*60
        M = 60
        
        h = int(duration / H)
        rm = duration % H
        m = int(rm / M)
        s = rm % M

        return f'{h}:{m}:{s:.2f}'

    def show_statistic(self):
        history = np.array(self._duration_from_latest_history) * 1000
        string = '[StopWatch] '
        string += f'avg: {history.mean()} ms | '
        string += f'median: {np.median(history)} ms | '
        string += f'max: {history.max()} ms | '
        string += f'min: {history.min()} ms | '
        string += f'cnt: {len(history)} |'
        return string
