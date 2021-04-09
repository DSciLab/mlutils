from torch.utils.data.dataset import Dataset as TorchDataset
from typing import TypeVar

T_co = TypeVar('T_co', covariant=True)


class Dataset(TorchDataset[T_co]):
    def update_transformer(self, *args, **kwargs):
        raise NotImplementedError
