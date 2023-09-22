from typing import Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ptls.frames import PtlsDataModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from src.const import PROJECT_NAME
from src.datamodules.autoencoder.dataset import MyColesDataset
from src.networks.decoders.base import AbsDecoder
from src.networks.modules.base import AbsAE
from src.preprocessing.churn_preproc import preprocessing
from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)


def train_autoencoder(
    cfg: DictConfig,
) -> None:
    mcc_column: str = cfg["dataset"]["mcc_column"]
    amt_column: str = cfg["dataset"]["amt_column"]

    dataset: List[Dict] = preprocessing(cfg["dataset"])

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
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        **cfg["trainer_args"],
    )

    trainer.fit(module, datamodule)
