from typing import TypeVar, Sequence, Optional
from torch.utils.data import Sampler
from torch.utils.data.dataloader import DataLoader as TorchDataLoader,\
                                        _collate_fn_t, _worker_init_fn_t
from .dataset import Dataset
from .log import Log

T_co = TypeVar('T_co', covariant=True)


class DataLoader(TorchDataLoader[T_co]):
    def __init__(self,
                 dataset: Dataset[T_co],
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 sampler: Optional[Sampler[int]] = None,
                 batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0,
                 worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None,
                 generator=None,
                 *,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False):

        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=sampler,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         drop_last=drop_last,
                         timeout=timeout,
                         worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context,
                         generator=generator,
                         prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers)

    def update_transformer(self, *args, **kwargs):
        try:
            self.dataset.update_transformer(*args, **kwargs)
        except NotImplementedError:
            Log.warn('Try to call a not implemented method \'update_transformer\'')

    def update_loss(self, *args, **kwds):
        try:
            self.batch_sampler.update_loss(*args, **kwds)
        except NotImplementedError:
            Log.warn('Try to call a not implemented method \'update_loss\'')
