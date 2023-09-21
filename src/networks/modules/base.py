import typing
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig
from hydra.utils import instantiate

from pytorch_lightning import LightningModule

import numpy as np
import pandas as pd

import torch
from torch import nn
from torcheval.metrics.functional import multiclass_f1_score, r2_score
from ptls.data_load import PaddedBatch
from ptls.frames.coles import ColesDataset
from ptls.frames import PtlsDataModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from src.networks.common_layers import PositionalEncoding
from src.networks.decoders import AbsDecoder
from src.utils.logging_utils import get_logger
from src.utils.metrtics import f1, r2

logger = get_logger(name=__name__)


class AbsAE(LightningModule):
    def __init__(
        self,
        encoder: SeqEncoderContainer,
        decoder: AbsDecoder,
        amnt_col: str,
        mcc_col: str,
        mcc_vocab_size: int,
        lr: float,
        weight_decay: float,
        encoder_weights: Optional[str] = "",
        decoder_weights: Optional[str] = "",
        unfreeze_enc_after: Optional[int] = 0,
        unfreeze_dec_after: Optional[int] = 0
    ) -> None:
        super().__init__()
        
        self.amnt_col = amnt_col
        self.mcc_col = mcc_col
        self.mcc_vocab_size = mcc_vocab_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.unfreeze_enc_after = unfreeze_enc_after
        self.unfreeze_dec_after = unfreeze_dec_after
        self.ae_output_size = decoder.output_size
        
        if encoder_weights:
            encoder.load_state_dict(torch.load(encoder_weights))
            
        if decoder_weights:
            decoder.load_state_dict(torch.load(decoder_weights))

        self.ae_core = nn.Sequential(encoder, decoder)
        
        if unfreeze_enc_after:
            logger.info("Freezing encoder weights")
            self.ae_core[0].requires_grad_(False)
            
        if unfreeze_dec_after:
            logger.info("Freezing decoder weights")
            self.ae_core[1].requires_grad_(False)
            
        
    def on_train_epoch_start(self) -> None:
        if self.unfreeze_enc_after and self.current_epoch == self.unfreeze_enc_after:
            logger.info("Unfreezing encoder weights")
            self.ae_core[0].requires_grad_(True)
            
        if self.unfreeze_dec_after and self.current_epoch == self.unfreeze_dec_after:
            logger.info("Unfreezing decoder weights")
            self.ae_core[1].requires_grad_(True)

        return super().on_train_epoch_start()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.lr,
            weight_decay=self.weight_decay,
        )
        
        return optimizer
