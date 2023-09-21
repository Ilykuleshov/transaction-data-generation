from typing import Dict, List

from omegaconf import DictConfig, OmegaConf

from ptls.frames import PtlsDataModule

from pytorch_lightning import Trainer

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
import wandb
from hydra.utils import instantiate

from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from src.preprocessing.churn_preproc import preprocessing
from src.datamodules.autoencoder.dataset import MyColesDataset
from src.networks.decoders.base import AbsDecoder
from src.utils.logging_utils import get_logger
from src.const import PROJECT_NAME
from src.networks.modules.base import AbsAE


logger = get_logger(name=__name__)


def train_autoencoder(
    cfg: DictConfig,
) -> None:
    mcc_column: str = cfg["dataset"]["mcc_column"]
    amt_column: str = cfg["dataset"]["amt_column"]

    dataset: List[Dict] = preprocessing(cfg["dataset"])

    with wandb.init(project=PROJECT_NAME, config=OmegaConf.to_container(cfg)):  # type: ignore
        train, val = train_test_split(dataset, test_size=0.2)

        datamodule = PtlsDataModule(
            train_data=MyColesDataset(train, cfg["dataset"]),
            valid_data=MyColesDataset(val, cfg["dataset"]),
            **cfg["datamodule_args"],
        )

        encoder: SeqEncoderContainer = instantiate(cfg["encoder"])
        decoder: AbsDecoder = instantiate(cfg["decoder"])
        module: AbsAE = instantiate(cfg["module"])(
            encoder=encoder,
            decoder=decoder,
            amnt_col=amt_column,
            mcc_col=mcc_column,
        )

        wandb_logger = WandbLogger(project=PROJECT_NAME)

        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            logger=wandb_logger,
            **cfg["trainer_args"],
        )

        trainer.fit(module, datamodule)
