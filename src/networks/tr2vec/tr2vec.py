import typing

import torch
from torch import nn

import pytorch_lightning as pl

from omegaconf import DictConfig

from utils.logging_utils import get_logger

logger = get_logger(name=__name__)


class Transaction2VecJoint(pl.LightningModule):

    def __init__(self, cfg: DictConfig, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(cfg)

        self.mcc_embedding_layer = nn.Embedding(cfg['mcc_vocab_size'] + 1, cfg['mcc_embed_size'], 0)
        self.mcc_output = nn.Linear(cfg['mcc_embed_size'], cfg['mcc_vocab_size'], False)
        self.mcc_criterion = nn.CrossEntropyLoss()


    def forward(self, ctx_mccs: torch.Tensor, ctx_lengths: torch.Tensor) -> torch.Tensor:
        mcc_hidden = self.mcc_embedding_layer(ctx_mccs) / ctx_lengths.view(-1, 1, 1)
        logits = self.mcc_output(mcc_hidden)
        return logits


    def configure_optimizers(self) -> typing.Mapping[str, typing.Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['learning_params']['lr'],
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            self.hparams['learning_params']['lr_scheduler_params']['factor'],
            self.hparams['learning_params']['lr_scheduler_params']['patience']
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        ctx_mccs, center_mccs, ctx_lengths = batch
        mcc_logits = self(ctx_mccs, ctx_lengths)
        loss = self.mcc_criterion(mcc_logits, center_mccs - 1)
        self.log('train_loss', loss, prog_bar=True)
        return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        ctx_mccs, center_mccs, ctx_lengths = batch
        mcc_logits = self(ctx_mccs, ctx_lengths)
        loss = self.mcc_criterion(mcc_logits, center_mccs - 1)
        self.log('val_loss', loss, prog_bar=True)
