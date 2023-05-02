import typing

from pytorch_lightning import LightningModule

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.networks.common_layers import PositionalEncoding
from src.utils.logging_utils import get_logger
from src.utils.metrtics import f1, r2, roc_auc

logger = get_logger(name=__name__)


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
        self.out_mcc    = nn.Linear(2 * embed_dim, n_vocab_size + 1)

        self.amount_loss_weights    = loss_weights[0]
        self.binary_loss_weights    = loss_weights[1]
        self.mcc_loss_weights       = loss_weights[2]

        self.mcc_criterion      = nn.CrossEntropyLoss(ignore_index=0)
        self.binary_criterion   = nn.BCELoss()
        self.amount_criterion   = nn.MSELoss()

        self.training_mcc_f1        = list()
        self.training_binary_f1     = list()
        self.training_binary_rocauc = list()
        self.training_amt_r2        = list()
        
        self.val_mcc_f1        = list()
        self.val_binary_f1     = list()
        self.val_binary_rocauc = list()
        self.val_amt_r2        = list()

        self.test_mcc_f1        = list()
        self.test_binary_f1     = list()
        self.test_binary_rocauc = list()
        self.test_amt_r2        = list()

    # Set pretrained tr2vec weights
    def set_embeds(self, mcc_weights: torch.Tensor):
        with torch.no_grad():
            self.mcc_embed.weight.data = mcc_weights
    
    def on_train_epoch_start(self) -> None:
        if self.hparams['freeze_embed'] \
            and self.current_epoch == self.hparams['unfreeze_after']:
            logger.info('Unfreezing embed weights')
            self.mcc_embed.requires_grad_(True)

        return super().on_train_epoch_start()

    def forward(
        self,
        user_id: torch.Tensor,
        mcc_codes: torch.Tensor,
        is_income: torch.Tensor,
        transaction_amt: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor 
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mcc_embed = self.mcc_embed(mcc_codes)
        mcc_embed = self.pe(mcc_embed)

        is_income = torch.unsqueeze(is_income, -1)
        transaction_amt = torch.unsqueeze(transaction_amt, -1)

        mat_orig = torch.cat((mcc_embed, transaction_amt, is_income), -1).float()
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
        seqs_after_lstm = pad_packed_sequence(packed_mat, True)[0]

        mcc_rec = self.out_mcc(seqs_after_lstm)
        is_income_rec = self.out_binary(seqs_after_lstm)
        amount_rec = self.out_amount(seqs_after_lstm)
        # for is_income we must maintain sigmoid layer by ourselfs
        # as BCEWithLogits will provide for all paddings .5 probability
        is_income_rec = torch.sigmoid(is_income_rec)

        mcc_rec, is_income_rec, amount_rec = self._trim_out_seq(
            [mcc_rec, is_income_rec, amount_rec],
            lengths
        )

        # squeeze for income and amount is required to reduce last 1 dimension
        return mcc_rec, is_income_rec.squeeze(), amount_rec.squeeze()

    def _compute_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        max_length = lengths.max().detach().cpu().item()
        mask = torch.arange(max_length) \
                .expand(len(lengths), max_length) \
                .to(self.device) < lengths.unsqueeze(1)
        mask.requires_grad_(False).unsqueeze_(-1)
        return mask

    # Method for returning original padding
    def _trim_out_seq(
        self,
        seqs_to_trim: tuple[torch.Tensor, ...],
        mask: torch.Tensor
    ) -> list[torch.Tensor]:
        return list(map(lambda seq: seq.masked_fill(~mask, 0), seqs_to_trim))
    
    def _calculate_metrics(
        
    ): ...
    
    def _calculate_losses(self, batch: tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.DoubleTensor,
        int,
        float
    ]) -> tuple[
            float,
            tuple[float, float, float],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ]:
        mcc_rec, is_income_rec, amount_rec = self(*batch[:-1])
        mcc_rec = torch.permute(mcc_rec, (0, 2, 1))
        
        mcc_loss = self.mcc_criterion(mcc_rec, batch[1])
        binary_loss = self.binary_criterion(is_income_rec, batch[2])
        amount_loss = self.amount_criterion(amount_rec, batch[3])

        total_loss = self.mcc_loss_weights * mcc_loss + \
            self.binary_loss_weights * binary_loss + \
            self.amount_loss_weights * amount_loss
        
        return (
            total_loss,
            (mcc_loss, binary_loss, amount_loss),
            (mcc_rec, is_income_rec, amount_rec)
        )
    
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
        loss, \
            (mcc_loss, binary_loss, amount_loss), \
            (mcc_rec, is_income_rec, amount_rec) = self._calculate_losses(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.log('train_loss_mcc', mcc_loss, on_step=True, prog_bar=False)
        self.log('train_loss_binary', binary_loss, on_step=True, prog_bar=False)
        self.log('train_loss_amt', amount_loss, on_step=True, prog_bar=False)
        
        mcc_rec = torch.argmax(mcc_rec, 1)
        mcc_rec = mcc_rec.flatten()[mcc_rec.flatten().nonzero()].squeeze()
        is_income_rec = is_income_rec.flatten()[
            is_income_rec.flatten().nonzero()
        ].squeeze()
        amount_rec = amount_rec.flatten()[amount_rec.flatten().nonzero()].squeeze()

        mcc_orig = batch[1].flatten()[batch[1].flatten().nonzero()].squeeze()
        is_income_orig = batch[2].flatten()[batch[2].flatten().nonzero()].squeeze()
        amount_orig = batch[3].flatten()[batch[3].flatten().nonzero()].squeeze()

        print(is_income_rec)
        print(is_income_orig)
        print(is_income_rec.shape)
        print(is_income_orig.shape)

        self.training_mcc_f1.append(f1(mcc_rec, mcc_orig, 'macro'))
        self.training_binary_f1.append(f1(is_income_rec >= .5, is_income_orig))
        self.training_binary_rocauc.append(roc_auc(is_income_rec, is_income_orig))
        self.training_amt_r2.append(r2(amount_rec, amount_orig))

        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log('train_mcc_f1', torch.stack(self.training_mcc_f1).mean())
        self.log('train_binary_f1', torch.stack(self.training_binary_f1).mean())
        self.log(
            'train_binary_ROCAUC',
            torch.stack(self.training_binary_rocauc).mean()
        )
        self.log('train_amt_r2', torch.stack(self.training_amt_r2).mean())

        self.training_mcc_f1.clear()
        self.training_binary_f1.clear()
        self.training_binary_rocauc.clear()
        self.training_amt_r2.clear()
    
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
        loss, \
            (mcc_loss, binary_loss, amount_loss), \
            (mcc_rec, is_income_rec, amount_rec) = self._calculate_losses(batch)
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
        loss, \
            (mcc_loss, binary_loss, amount_loss), \
            (mcc_rec, is_income_rec, amount_rec) = self._calculate_losses(batch)
        self.log('test_loss', loss, on_epoch=True, on_step=False)

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
        return [opt], [{'scheduler': scheduler, 'monitor': 'val_loss'}]

