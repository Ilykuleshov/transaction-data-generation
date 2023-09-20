import typing
from typing import Any, Dict, Optional, Tuple

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

from src.networks.autoencoder_core import CoreBase
from src.networks.common_layers import PositionalEncoding
from src.utils.logging_utils import get_logger
from src.utils.metrtics import f1, r2

logger = get_logger(name=__name__)


class LSTMAE(LightningModule):
    def __init__(
        self,
        n_mcc_codes: int,
        loss_weights: Dict,
        freeze_embed: bool,
        unfreeze_after: int,
        lr: float,
        weight_decay: float,
        amnt_col: str,
        mcc_col: str,
        core_ae: DictConfig,
        *args: typing.Any,
        weights_path: Optional[str] = "",
        **kwargs: typing.Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(
            {
                "n_mcc_codes": n_mcc_codes,
                "loss_weights": loss_weights,
                "freeze_embed": freeze_embed,
                "unfreeze_after": unfreeze_after,
                "lr": lr,
                "weight_decay": weight_decay,
                "amnt_col": amnt_col,
                "mcc_col": mcc_col,
                "core_ae": core_ae,
            }
        )

        self.ae_core: CoreBase = instantiate(core_ae)
        if weights_path:
            self.ae_core.encoder.load_state_dict(
                torch.load(weights_path)
            )

        self.out_amount = nn.Linear(self.ae_core.output_size, 1)
        self.out_mcc = nn.Linear(self.ae_core.output_size, n_mcc_codes + 1)

        self.amount_loss_weights = loss_weights["amount"] / sum(loss_weights.values())
        self.mcc_loss_weights = loss_weights["mcc"] / sum(loss_weights.values())

        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.amount_criterion = nn.MSELoss()

        self.training_mcc_f1 = list()
        self.training_amt_r2 = list()

        self.val_mcc_f1 = list()
        self.val_amt_r2 = list()

        self.test_mcc_f1 = list()
        self.test_amt_r2 = list()

    def on_train_epoch_start(self) -> None:
        if (
            self.hparams["freeze_embed"]
            and self.current_epoch == self.hparams["unfreeze_after"]
        ):
            logger.info("Unfreezing embed weights")
            self.ae_core.encoder.requires_grad_(True)

        return super().on_train_epoch_start()

    def forward(
        self,
        batch: PaddedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seqs_after_lstm = self.ae_core(batch)  # supposedly (B * S, L, E)
        mcc_rec = self.out_mcc(seqs_after_lstm)
        amount_rec = self.out_amount(seqs_after_lstm)

        # zero-out padding to disable grad flow
        mcc_rec[~batch.seq_len_mask] = 0
        amount_rec[~batch.seq_len_mask] = 0

        # squeeze for amount is required to reduce last dimension
        return (mcc_rec, amount_rec.squeeze(dim=-1))

    def _calculate_metrics(
        self,
        mcc_preds: torch.Tensor,
        amt_value: torch.Tensor,
        mcc_orig: torch.Tensor,
        amt_orig: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[float, float]:
        with torch.no_grad():
            # print(mcc_probs.shape)
            # print(mask.shape)

            mcc_orig = mcc_orig[mask].flatten()
            mcc_preds = mcc_preds[mask].reshape((*mcc_orig.shape, -1))

            mask.squeeze_()
            return (
                multiclass_f1_score(
                    mcc_preds,
                    mcc_orig,
                    average="macro",
                    num_classes=self.hparams["n_mcc_codes"] + 1,
                ).item(),
                r2_score(amt_value[mask].flatten(), amt_orig[mask].flatten()).item(),
            )

    def _calculate_losses(
        self,
        mcc_rec: torch.Tensor,
        amount_rec: torch.Tensor,
        mcc_orig: torch.Tensor,
        amount_orig: torch.Tensor,
    ) -> tuple[float, tuple[float, float]]:
        # Lengths tensor

        mcc_loss = self.mcc_criterion(mcc_rec.transpose(2, 1), mcc_orig)
        amount_loss = torch.log(self.amount_criterion(amount_rec, amount_orig))

        total_loss = (
            self.mcc_loss_weights * mcc_loss + self.amount_loss_weights * amount_loss
        )

        return (total_loss, (mcc_loss, amount_loss))

    def _all_forward_step(self, batch: Tuple[PaddedBatch, torch.Tensor]):
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

    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
    #     mask = compute_mask(batch[-2], self.device)
    #     mcc_rec, is_income_rec, amount_rec = self(*batch[:-1], mask)
    #     total_loss, (mcc_loss, amount_loss) = self._calculate_losses(
    #         mcc_rec, is_income_rec, amount_rec, *batch[1:4]
    #     )

    #     return(
    #         total_loss,
    #         (mcc_loss, amount_loss),
    #         (mcc_rec, is_income_rec, amount_rec)
    #     )

    def training_step(
        self, batch: Tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> float:
        loss, (mcc_loss, amount_loss), (f1_mcc, r2_amount) = self._all_forward_step(
            batch
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_loss_mcc", mcc_loss, on_step=True, prog_bar=False)
        self.log("train_loss_amt", amount_loss, on_step=True, prog_bar=False)

        self.training_mcc_f1.append(f1_mcc)
        self.training_amt_r2.append(r2_amount)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_mcc_f1", float(np.mean(self.training_mcc_f1)))
        self.log("train_amt_r2", float(np.mean(self.training_amt_r2)))

        self.training_mcc_f1.clear()
        self.training_amt_r2.clear()

    def validation_step(
        self, batch: Tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> None:
        loss, (mcc_loss, amount_loss), (f1_mcc, r2_amount) = self._all_forward_step(
            batch
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss_mcc", mcc_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val_loss_amt", amount_loss, on_step=False, on_epoch=True, prog_bar=False
        )

        self.val_mcc_f1.append(f1_mcc)
        self.val_amt_r2.append(r2_amount)

    def on_validation_epoch_end(self) -> None:
        self.log("val_mcc_f1", float(np.mean(self.val_mcc_f1)))
        self.log("val_amt_r2", float(np.mean(self.val_amt_r2)))

        self.val_mcc_f1.clear()
        self.val_amt_r2.clear()

    def test_step(
        self, batch: Tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ) -> None:
        loss, (mcc_loss, amount_loss), (f1_mcc, r2_amount) = self._all_forward_step(
            batch
        )
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_loss_mcc", mcc_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test_loss_amt", amount_loss, on_step=False, on_epoch=True, prog_bar=False
        )

        self.test_mcc_f1.append(f1_mcc)
        self.test_amt_r2.append(r2_amount)

    def on_test_epoch_end(self) -> None:
        self.log("test_mcc_f1", float(np.mean(self.test_mcc_f1)))
        self.log("test_amt_r2", float(np.mean(self.test_amt_r2)))

        self.test_mcc_f1.clear()
        self.test_amt_r2.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", 1e-1, 2, verbose=True
        )
        return [opt], [{"scheduler": scheduler, "monitor": "val_loss"}]
