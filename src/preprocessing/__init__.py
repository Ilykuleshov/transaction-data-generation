import typing

from omegaconf import DictConfig

from .gender_preprop import preprocessing as g_preprop
from .new_data_preprop import preprocessing as n_preprop


def preprocessing(name: str) -> typing.Callable:
    if name == 'new_data':
        return n_preprop
    elif name =='gender':
        return g_preprop
