import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import CoreBase
from src.utils.data_utils import trim_out_seq, compute_mask


class NLPLSTM(CoreBase):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        *args, **kwargs
    ) -> None:
        super().__init__(input_size, hidden_size, output_size, *args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

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

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=output_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.proj = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, data: Tensor, lengths: Tensor) -> Tensor:
        B = data.shape[0]
        L = data.shape[1]
        
        packed_mat = pack_padded_sequence(
            data,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_mat, _ = self.encoder1(packed_mat)
        packed_mat, hidden_state = self.encoder2(packed_mat)

        decoder_input = torch.zeros(B, 1, self.hidden_size, device=data.device)
        decoder_outputs = torch.zeros(B, L, self.output_size, device=data.device)
        
        for i in range(L):
            decoder_output, hidden_state = self.decoder(
                decoder_input, hidden_state
            )
            decoder_outputs[:, i, :] = decoder_output.clone().squeeze(1)
            decoder_input = self.proj(decoder_output)
        
        output = trim_out_seq(decoder_outputs, compute_mask(
            lengths, data.device
        ))

        return output
