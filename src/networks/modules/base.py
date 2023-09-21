from typing import Any, Optional, Union


from pytorch_lightning import LightningModule


import torch
from torch import nn, Tensor

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load import PaddedBatch

from src.networks.decoders.base import AbsDecoder
from src.utils.logging_utils import get_logger

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
        unfreeze_dec_after: Optional[int] = 0,
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

        self.encoder = encoder
        self.decoder = decoder

        if unfreeze_enc_after:
            logger.info("Freezing encoder weights")
            self.encoder.requires_grad_(False)

        if unfreeze_dec_after:
            logger.info("Freezing decoder weights")
            self.decoder.requires_grad_(False)

    def forward(self, x: Union[Tensor, PaddedBatch]) -> Any:
        embeddings = self.encoder(x)
        if isinstance(embeddings, PaddedBatch):
            embeddings = embeddings.payload

        return self.decoder(embeddings)

    def on_train_epoch_start(self) -> None:
        if self.unfreeze_enc_after and self.current_epoch == self.unfreeze_enc_after:
            logger.info("Unfreezing encoder weights")
            self.encoder.requires_grad_(True)

        if self.unfreeze_dec_after and self.current_epoch == self.unfreeze_dec_after:
            logger.info("Unfreezing decoder weights")
            self.decoder.requires_grad_(True)

        return super().on_train_epoch_start()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.lr,
            weight_decay=self.weight_decay,
        )

        return optimizer
