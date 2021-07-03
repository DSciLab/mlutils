from typing import Union
import numpy as np


ALLOW_UNIT = ['B', 'K', 'M', 'G']


class CachedLoader(object):
    ESP = 1.0e-7

    def __init__(self, max_size: int, unit: str='M') -> None:
        super().__init__()
        assert unit in ALLOW_UNIT, f'Unit ({unit}) is not available.'
        self.max_cache_size_B = unit_translate_to_B(max_size, unit)
        self.cached_size_B = 0
        self._cache = {}
        self._cache_hit_cnt = {}
        self.hit_cnt = 0
        self.read_cnt = 0

    def __str__(self) -> str:
        mem_size = unit_translate_from_B(self.cached_size_B, 'G')
        string = f'<CachedLoader length: {len(self)}, '\
                 f'memory size: {mem_size:.3f} GB, hit rate: {self.hit_rate:.3f}>'
        return string

    @property
    def hit_rate(self) -> float:
        return self.hit_cnt / (self.read_cnt + self.ESP)

    def del_entry(self, npy_path: str) -> None:
        if self.in_cache(npy_path):
            data = self._cache[npy_path]
            self.cached_size_B -= sizeof(data)
            del self._cache[npy_path]
            del self._cache_hit_cnt[npy_path]

    remove = del_entry

    def maybe_load_from_cache(
        self,
        npy_path: str
    ) -> Union[None, np.ndarray]:
        if self.in_cache(npy_path):
            # print(self)
            self.hit_cnt += 1
            self._cache_hit_cnt[npy_path] += 1
            return self._cache[npy_path]
        else:
            return None

    def in_cache(self, npy_path: str) -> bool:
        return npy_path is self._cache_hit_cnt.keys()

    @property
    def full(self) -> bool:
        return self.cached_size_B >= self.max_cache_size_B

    def __len__(self) -> int:
        return len(self._cache)

    def maybe_cache_data(
        self,
        data: np.ndarray,
        npy_path: str
    ) -> None:
        # print(f'cache {npy_path}, {self._cache.keys()}, {self.full}, {self.in_cache(npy_path)}')
        if self.full or self.in_cache(npy_path):
            return
        self.cached_size_B += sizeof(data)
        self._cache_hit_cnt[npy_path] = 0
        self._cache[npy_path] = data

    def load_from_disk(
        self,
        npy_path: str
    ) -> np.ndarray:
        return np.load(npy_path)

    def load_npy(self, npy_path: str) -> np.ndarray:
        # most recent hit rate
        if self.read_cnt > 1000:
            self.read_cnt = 0
            self.hit_cnt = 0
        self.read_cnt += 1
        data = self.maybe_load_from_cache(npy_path)
        if data is None:
            # load from disk
            # and maybe cache data
            data = self.load_from_disk(npy_path)
            self.maybe_cache_data(data, npy_path)
        return data


def sizeof(inp: np.ndarray) -> int:
    """
    :return: number of byte
    """
    return inp.itemsize * inp.size


def unit_translate_to_B(size: int, unit: str) -> Union[int, float]:
    """
    :return: number of byte
    """
    if unit == 'B':
        return size
    elif unit == 'K':
        return size * 1024
    elif unit == 'M':
        return size * 1048576
    elif unit == 'G':
        return size * 1073741824
    else:
        raise ValueError(f'Unit ({unit}) is not available.')


def unit_translate_from_B(size: int, unit: str) -> Union[int, float]:
    if unit == 'B':
        return size
    elif unit == 'K':
        return size / 1024
    elif unit == 'M':
        return size / 1048576
    elif unit == 'G':
        return size / 1073741824
    else:
        raise ValueError(f'Unit ({unit}) is not available.')
