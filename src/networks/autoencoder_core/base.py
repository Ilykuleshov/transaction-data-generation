from abc import ABC

import torch
import torch.nn as nn


class CoreBase(ABC, nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    
    def forward(
        self, data: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def output_size(self):
        raise NotImplementedError()
    