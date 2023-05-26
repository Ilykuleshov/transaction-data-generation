import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import CoreBase


class SimpleLSTM(CoreBase):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        *args, **kwargs
    ) -> None:
        super().__init__(input_size, hidden_size, output_size, *args, **kwargs)

        self.encoder1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first = True
        )
        self.encoder2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first = True
        )

        self.decoder1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first = True
        )
        self.decoder2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=output_size,
            num_layers=num_layers,
            batch_first = True
        )

    def forward(self, data: Tensor, lengths: Tensor) -> Tensor:
        packed_mat = pack_padded_sequence(
            data,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_mat, _ = self.encoder1(packed_mat)
        packed_mat, _ = self.encoder2(packed_mat)

        packed_mat, _ = self.decoder1(packed_mat)
        packed_mat, _ = self.decoder2(packed_mat)

        seqs_after_lstm = pad_packed_sequence(packed_mat, True)[0]
        return seqs_after_lstm
