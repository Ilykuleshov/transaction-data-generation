from typing import Dict, Optional, Tuple, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch import nn, Tensor
from torcheval.metrics.functional import multiclass_auroc, r2_score
from ptls.data_load import PaddedBatch
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from src.networks.decoders.base import AbsDecoder
from src.networks.modules.base import AbsAE
from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)


class VanillaAE(AbsAE):
    def __init__(
        self,
        encoder: SeqEncoderContainer,
        decoder: AbsDecoder,
        amnt_col: str,
        mcc_col: str,
        mcc_vocab_size: int,
        loss_weights: Dict,
        **kwargs,
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            amnt_col=amnt_col,
            mcc_col=mcc_col,
            mcc_vocab_size=mcc_vocab_size,
            **kwargs,
        )

        self.out_amount = nn.Linear(self.ae_output_size, 1)
        self.out_mcc = nn.Linear(self.ae_output_size, mcc_vocab_size + 1)

        self.amount_loss_weights = loss_weights["amount"] / sum(loss_weights.values())
        self.mcc_loss_weights = loss_weights["mcc"] / sum(loss_weights.values())

        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.amount_criterion = nn.MSELoss()

    def forward(
        self,
        batch: PaddedBatch,
    ) -> tuple[Tensor, Tensor]:
        seqs_after_lstm = super().forward(batch)  # supposedly (B * S, L, E)

        mcc_rec = self.out_mcc(seqs_after_lstm)
        amount_rec = self.out_amount(seqs_after_lstm)

        # zero-out padding to disable grad flow
        mcc_rec[~batch.seq_len_mask] = 0
        amount_rec[~batch.seq_len_mask] = 0

        # squeeze for amount is required to reduce last dimension
        return (mcc_rec, amount_rec.squeeze(dim=-1))

    def _calculate_metrics(
        self,
        mcc_preds: Tensor,
        amt_value: Tensor,
        mcc_orig: Tensor,
        amt_orig: Tensor,
        mask: Tensor,
    ) -> tuple[float, float]:
        with torch.no_grad():
            mcc_orig = mcc_orig[mask].flatten()
            mcc_preds = mcc_preds[mask].reshape((*mcc_orig.shape, -1))

            mask.squeeze_()
            return (
                multiclass_auroc(
                    mcc_preds,
                    mcc_orig,
                    num_classes=self.mcc_vocab_size + 1,
                ).item(),
                r2_score(amt_value[mask].flatten(), amt_orig[mask].flatten()).item(),
            )

    def _calculate_losses(
        self,
        mcc_rec: Tensor,
        amount_rec: Tensor,
        mcc_orig: Tensor,
        amount_orig: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # Lengths tensor

        mcc_loss = self.mcc_criterion(mcc_rec.transpose(2, 1), mcc_orig)
        amount_loss = torch.log(self.amount_criterion(amount_rec, amount_orig))

        total_loss = (
            self.mcc_loss_weights * mcc_loss + self.amount_loss_weights * amount_loss
        )

        return (total_loss, (mcc_loss, amount_loss))

    def _all_forward_step(self, batch: Tuple[PaddedBatch, Tensor]):
        padded_batch = batch[0]
        mcc_rec, amount_rec = self(padded_batch)  # (B * S, L, MCC_N), (B * S, L)
        mcc_orig = padded_batch.payload["mcc_code"]
        amount_orig = padded_batch.payload["amount"]

        total_loss, (mcc_loss, amount_loss) = self._calculate_losses(
            mcc_rec, amount_rec, mcc_orig, amount_orig
        )

        f1_mcc, r2_amount = self._calculate_metrics(
            mcc_rec, amount_rec, mcc_orig, amount_orig, padded_batch.seq_len_mask
        )

        return (total_loss, (mcc_loss, amount_loss), (f1_mcc, r2_amount))

    def _step(self, stage: str, batch: Tuple[PaddedBatch, Tensor], batch_idx: int, *args, **kwargs):
        if not self.trainer:
            raise ValueError("No trainer!")


        loss, (mcc_loss, amount_loss), (f1_mcc, r2_amount) = self._all_forward_step(
            batch
        )
        
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        self.log(f"{stage}_loss_mcc", mcc_loss, on_step=True, prog_bar=False)
        self.log(f"{stage}_loss_amt", amount_loss, on_step=True, prog_bar=False)

        self.log(f"{stage}_mcc_f1", f1_mcc, on_step=False, on_epoch=True)
        self.log(f"{stage}_amt_r2", r2_amount, on_step=False, on_epoch=True)

        return loss

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return self._step("train", *args, **kwargs)
    
    def validation_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Union[STEP_OUTPUT, None]:
        return self._step("test", *args, **kwargs)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", 1e-1, 2, verbose=True
        )
        return [opt], [{"scheduler": scheduler, "monitor": "val_loss"}]
