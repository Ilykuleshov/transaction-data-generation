import typing
import time
from datetime import timedelta

from pytorch_lightning import LightningModule

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.networks.common_layers import PositionalEncoding


class LSTMAE(LightningModule):

    def __init__(
        self,
        n_features: int,
        embed_dim: int,
        num_layers: int,
        mcc_embed_dim: int,
        n_vocab_size: int,
        *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters({
            'n_features'    : n_features,
            'embed_dim'     : embed_dim,
            'num_layers'    : num_layers,
            'mcc_embed_dim' : mcc_embed_dim,
            'n_vocab_size'  : n_vocab_size
        })

        self.mcc_embed = nn.Embedding(n_vocab_size + 1, mcc_embed_dim, padding_idx=0)
        self.pe = PositionalEncoding(mcc_embed_dim)

        self.encoder1 = nn.LSTM(
            input_size=n_features,
            hidden_size=embed_dim * 2,
            num_layers=num_layers,
            batch_first=True
        )
        self.encoder2 = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder1 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim * 2,
            num_layers=1,
            batch_first=True
        )
        self.decoder2 = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=embed_dim * 2,
            num_layers=1,
            batch_first=True
        )

        self.out_amount = nn.Linear(2 * embed_dim, 1)
        self.out_binary = nn.Linear(2 * embed_dim, 1)
        self.out_mcc    = nn.Linear(2 * embed_dim, n_vocab_size)

        self.train_time: float = None

    # Set pretrained tr2vec weights
    def set_embeds(self, mcc_weights: torch.Tensor):
        with torch.no_grad():
            self.mcc_embed.weight.data = mcc_weights
    
    def on_train_epoch_start(self) -> None:
        self.train_time = time.time()
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        train_time = time.time() - self.train_time
        train_time = str(timedelta(seconds=train_time))

        self.log('train_time', train_time, prog_bar=True, on_step=False, on_epoch=True)
        return super().on_train_epoch_end()

    def forward(
        self,
        user_id: torch.Tensor,
        mcc_codes: torch.Tensor,
        is_income: torch.Tensor,
        transaction_amt: torch.Tensor,
        lengths: torch.Tensor
    ) -> None:
        mcc_embed = self.mcc_embed(mcc_codes)
        mcc_embed = self.pe(mcc_embed)

        is_income = torch.unsqueeze(is_income, -1)
        transaction_amt = torch.unsqueeze(transaction_amt, -1)

        mat_orig = torch.cat((mcc_embed, transaction_amt, is_income), -1)
        # Pack sequnces to get a real hidden state no matter of difference in lengths
        packed_mat = pack_padded_sequence(
            mat_orig,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_mat, _ = self.encoder1(packed_mat)
        packed_mat, _ = self.encoder2(packed_mat)

        packed_mat, _ = self.decoder1(packed_mat)
        packed_mat, _ = self.decoder2(packed_mat)

        # Get original format (batch_size X 2 * embed_dim X max_length)
        seqs_after_lstm = pad_packed_sequence(packed_mat, True)

        mcc_rec = self.out_mcc(seqs_after_lstm)
        is_income_rec = self.out_binary(seqs_after_lstm)
        amount_rec = self.out_amount(seqs_after_lstm)

        mcc_rec, is_income_rec, amount_rec = self._trim_out_seq(
            [mcc_rec, is_income_rec, amount_rec]
        )

        return mcc_rec, is_income_rec, amount_rec

    # Method for returning original padding
    @staticmethod
    def _trim_out_seq(
        seqs_to_trim: typing.List[torch.Tensor],
        lengths: torch.Tensor
    ) -> typing.List[torch.Tensor]:
        max_length = seqs_to_trim[0].shape[1]
        mask = torch.arange(max_length) \
                .expand(len(lengths), max_length) < lengths.unsqueeze(1)
        return list(map(lambda seq: seq.masked_fill_(~mask), seqs_to_trim))
