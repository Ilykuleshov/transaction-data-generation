from abc import ABC

import torch
import torch.nn as nn


class CoreBase(ABC, nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self, data: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()
