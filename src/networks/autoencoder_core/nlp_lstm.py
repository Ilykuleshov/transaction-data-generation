from typing import List
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load import PaddedBatch

from .base import CoreBase
from src.utils.data_utils import trim_out_seq, compute_mask


class NLPLSTM(CoreBase):
    def __init__(
        self,
        encoder: SeqEncoderContainer,
        hidden_size: int,
        proj_size: int,
        num_layers: int
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.decoder = nn.LSTM(
            input_size=encoder.embedding_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.embedding_size = encoder.embedding_size
        self.proj_size = proj_size

    def forward(self, data: PaddedBatch) -> Tensor:
        embeddings_batch: PaddedBatch = self.encoder(data) # (B * S, L, E)
        decoded_batch = self.decoder(embeddings_batch.payload)[0] # (B * S, L, P)

        return decoded_batch # (B * S, L, E)
    
    @property
    def output_size(self):
        return self.proj_size
