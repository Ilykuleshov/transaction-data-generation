from typing import Tuple, Optional
import torch
from torch import nn, Tensor
from src.networks.decoders.base import AbsDecoder

class LSTMCellDecoder(AbsDecoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.output_size = hidden_size
        
    def forward(self, input: Tensor, L: int, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        B = input.shape[0]
        H = self.output_size
        hidden_state, cell_state = hx or (input.new_zeros(B, H), input.new_zeros(B, H))
        outputs_list = []
        for _ in range(L):
            hidden_state, cell_state = self.cell(input, (hidden_state, cell_state))
            outputs_list.append(input)
            input = cell_state

        return torch.stack(outputs_list, dim=1)
    