import os
import pickle
from typing import Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf
import pandas as pd

from ptls.frames import PtlsDataModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
import wandb

from ptls.data_load.datasets import ParquetDataset

from src.preprocessing.churn_preproc import preprocessing
from src.datamodules.autoencoder.dataset import MyColesDataset
from src.utils.logging_utils import get_logger
from src.const import PROJECT_NAME
from src.networks.lstm import LSTMAE


logger = get_logger(name=__name__)


def train_autoencoder(cfg_preprop: DictConfig, cfg_model: DictConfig) -> None:
    cfg = OmegaConf.merge({"dataset": cfg_preprop, "model": cfg_model})
    dir_path: str = cfg_preprop["dir_path"]
    mcc_column: str = cfg_preprop["mcc_column"]
    amt_column: str = cfg_preprop["amt_column"]

    dataset: List[Dict] = preprocessing(cfg_preprop)

    n_mccs: int = cfg_model["n_mccs"]
    freeze_embed: bool = cfg_model["freeze_embed"]
    unfreeze_after: int = cfg_model["unfreeze_after"]
    core_ae: DictConfig = cfg_model["autoencoder_core"]
    loss_weights: Dict = cfg_model["loss_weights"]
    weights_path: str = cfg_model["weights_path"]

    lr = cfg_model["learning_params"]["lr"]
    weight_decay = cfg_model["learning_params"]["weight_decay"]
    # lr_schedule_params = cfg_model["learning_params"]["lr_schedule_params"]
    # max_epochs = cfg_model["learning_params"]["max_epochs"]
    # early_stopping_params = cfg_model["learning_params"]["early_stopping_params"]

    dir_coles = os.path.join(dir_path, "coles")

    with wandb.init(project=PROJECT_NAME, config=OmegaConf.to_container(cfg)): # type: ignore
        train, val = train_test_split(dataset, test_size=0.2)

        training_params: DictConfig = cfg_model["learning_params"]

        datamodule = PtlsDataModule(
            train_data=MyColesDataset(train, cfg_preprop),
            train_batch_size=training_params["train_batch_size"],
            train_num_workers=training_params["train_num_workers"],
            valid_data=MyColesDataset(val, cfg_preprop),
            valid_batch_size=training_params["val_batch_size"],
            valid_num_workers=training_params["val_num_workers"],
        )

        model = LSTMAE(
            n_mcc_codes=n_mccs,
            freeze_embed=freeze_embed,
            unfreeze_after=unfreeze_after,
            core_ae=core_ae,
            loss_weights=loss_weights,
            lr=lr,
            weight_decay=weight_decay,
            amnt_col=amt_column,
            mcc_col=mcc_column,
            weights_path=weights_path
        )

        wandb_logger = WandbLogger(
            project=PROJECT_NAME
        )

        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=training_params["epochs"],
            log_every_n_steps=20,
            logger=wandb_logger,
            accumulate_grad_batches=training_params["accum_grad_batches"]
        )

        trainer.fit(model, datamodule)
