import typing
import time
from datetime import timedelta

from pytorch_lightning import LightningModule

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.networks.common_layers import PositionalEncoding


class LSTMAE(LightningModule):

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        mcc_embed_dim: int,
        n_vocab_size: int,
        loss_weights: tuple[float, float, float],
        freeze_embed: bool,
        unfreeze_after: int,
        lr: float,
        weight_decay: float,
        *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters({
            'embed_dim'     : embed_dim,
            'num_layers'    : num_layers,
            'mcc_embed_dim' : mcc_embed_dim,
            'n_vocab_size'  : n_vocab_size,
            'loss_weights'  : loss_weights,
            'freeze_embed'  : freeze_embed,
            'unfreeze_after': unfreeze_after,
            'lr'            : lr,
            'weight_decay'  : weight_decay
        })

        n_features = mcc_embed_dim + 2

        self.mcc_embed = nn.Embedding(
            n_vocab_size + 1,
            mcc_embed_dim,
            padding_idx=0
        )
        if freeze_embed:
            with torch.no_grad():
                self.mcc_embed.requires_grad_(False)

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

        self.amount_loss_weights    = loss_weights[0]
        self.binary_loss_weights    = loss_weights[1]
        self.mcc_loss_weights       = loss_weights[2]

        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.binary_criterion = nn.BCELoss()
        self.amount_criterion = nn.MSELoss()

        self.train_time: float = None

    # Set pretrained tr2vec weights
    def set_embeds(self, mcc_weights: torch.Tensor):
        with torch.no_grad():
            self.mcc_embed.weight.data = mcc_weights
    
    def on_train_epoch_start(self) -> None:
        self.train_time = time.time()
        if self.hparams['freeze_embed'] \
            and self.current_epoch == self.hparams['unfreeze_after']:
            self.mcc_embed.requires_grad_(True)

        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        train_time = time.time() - self.train_time
        train_time = str(timedelta(seconds=train_time))

        self.log(
            'train_time',
            train_time,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        return super().on_train_epoch_end()

    def forward(
        self,
        user_id: torch.Tensor,
        mcc_codes: torch.Tensor,
        is_income: torch.Tensor,
        transaction_amt: torch.Tensor,
        lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mcc_embed = self.mcc_embed(mcc_codes)
        mcc_embed = self.pe(mcc_embed)

        is_income = torch.unsqueeze(is_income, -1)
        transaction_amt = torch.unsqueeze(transaction_amt, -1)

        mat_orig = torch.cat((mcc_embed, transaction_amt, is_income), -1)
        # Pack sequnces to get a real hidden state
        # no matter of difference in lengths
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

        # for is_income we must maintain sigmoid layer by ourselfs
        # as BCEWithLogits will provide for all paddings .5 probability
        is_income_rec = F.sigmoid(is_income_rec)

        mcc_rec, is_income_rec, amount_rec = self._trim_out_seq(
            [mcc_rec, is_income_rec, amount_rec]
        )

        # squeeze for income and amount is required to reduce last 1 dimension
        return mcc_rec, is_income_rec.squeeze(), amount_rec.squeeze()

    # Method for returning original padding
    @staticmethod
    def _trim_out_seq(
        seqs_to_trim: tuple[torch.Tensor, ...],
        lengths: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        max_length = seqs_to_trim[0].shape[1]
        mask = torch.arange(max_length) \
                .expand(len(lengths), max_length) < lengths.unsqueeze(1)
        return list(map(lambda seq: seq.masked_fill_(~mask), seqs_to_trim))
    
    def _calculate_losses(self, batch: tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.DoubleTensor,
        int,
        float
    ]) -> float:
        mcc_rec, is_income_rec, amount_rec = self(*batch[:-1])
        mcc_rec = torch.permute(mcc_rec, (0, 2, 1))
        
        mcc_loss = self.mcc_criterion(mcc_rec, batch[1])
        binary_loss = self.binary_criterion(is_income_rec, batch[2])
        amount_loss = self.amount_criterion(amount_rec, batch[3])

        return self.mcc_loss_weights * mcc_loss + \
            self.binary_loss_weights * binary_loss + \
            self.amount_loss_weights * amount_loss
    
    def training_step(
        self,
        batch: tuple[
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.DoubleTensor,
            int,
            float
        ],
        batch_idx: int
    ) -> typing.Mapping[str, float]:
        loss = self._calculate_losses(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True)

        return {'loss': loss}
    
    def validation_step(
        self,
        batch: tuple[
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.DoubleTensor,
            int,
            float
        ],
        batch_idx: int
    ) -> None:
        loss = self._calculate_losses(batch)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(
        self,
        batch: tuple[
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.DoubleTensor,
            int,
            float
        ],
        batch_idx: int
    ) -> None:
        self.log(
            'test_loss',
            self._calculate_losses(batch),
            on_epoch=True,
            on_step=False
        )

    def configure_optimizers(self) -> typing.Mapping[str, typing.Any]:
        opt = torch.optim.AdamW(
            self.parameters(),
            self.hparams['lr'],
            weight_decay=self.hparams['weight_decay']
        )
        scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            'min',
            1e-1,
            2,
            verbose=True
        )
        return {'optimizer': opt, 'scheduler': scheduler, 'monitor': 'val_loss'}

