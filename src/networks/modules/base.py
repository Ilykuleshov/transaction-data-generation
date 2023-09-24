from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


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
        optimizer_config: DictConfig,
        encoder_weights: Optional[str] = "",
        decoder_weights: Optional[str] = "",
        unfreeze_enc_after: Optional[int] = 0,
        unfreeze_dec_after: Optional[int] = 0,
    ) -> None:
        super().__init__()

        self.amnt_col = amnt_col
        self.mcc_col = mcc_col
        self.mcc_vocab_size = mcc_vocab_size
        self.unfreeze_enc_after = unfreeze_enc_after
        self.unfreeze_dec_after = unfreeze_dec_after
        self.ae_output_size = decoder.output_size
        self.optimizer_config = optimizer_config  # type: ignore
        self.lr = optimizer_config["optimizer"]["lr"]

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

    def forward(self, x: PaddedBatch) -> Any:
        embeddings: Union[PaddedBatch, Tensor] = self.encoder(x)
        if isinstance(embeddings, PaddedBatch):
            return self.decoder(embeddings.payload)
        elif embeddings.ndim == 3:
            return self.decoder(embeddings)
        elif embeddings.ndim == 2:
            return self.decoder(embeddings, x.seq_feature_shape[1])
        else:
            raise ValueError(f"")

    def on_train_epoch_start(self) -> None:
        if self.unfreeze_enc_after and self.current_epoch == self.unfreeze_enc_after:
            logger.info("Unfreezing encoder weights")
            self.encoder.requires_grad_(True)

        if self.unfreeze_dec_after and self.current_epoch == self.unfreeze_dec_after:
            logger.info("Unfreezing decoder weights")
            self.decoder.requires_grad_(True)

        return super().on_train_epoch_start()

    def configure_optimizers(self):
        cnf: Dict = instantiate(self.optimizer_config, _convert_="all")
        cnf["optimizer"] = cnf["optimizer"](
            lr=self.lr, params=self.parameters()
        )

        if (scheduler := cnf.get("lr_scheduler", None)):
            if isinstance(scheduler, dict):
                scheduler["scheduler"] = scheduler["scheduler"](optimizer=cnf["optimizer"])
            else:
                cnf["lr_scheduler"] = scheduler(optimizer=cnf["optimizer"])

        return cnf
