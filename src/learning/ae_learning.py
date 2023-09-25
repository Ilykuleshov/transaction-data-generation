from typing import Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split, DataLoader
from ptls.frames import PtlsDataModule
from ptls.data_load.datasets.memory_dataset import MemoryMapDataset
from ptls.data_load.datasets.augmentation_dataset import AugmentationDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load.augmentations import RandomSlice
from ptls.data_load.utils import collate_feature_dict
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger

from src.const import PROJECT_NAME
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

    dataset_list: List[Dict] = preprocessing(cfg["dataset"])
    
    dataset = AugmentationDataset(
        MemoryMapDataset(dataset_list, [SeqLenFilter(cfg["dataset"]["min_len"])]),
        [RandomSlice(cfg["dataset"]["max_len"], cfg["dataset"]["max_len"])]
    )

    train, val = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(
        train, collate_fn=collate_feature_dict, **cfg.get("train_dl_args", {})
    )

    val_dataloader = DataLoader(
        val, collate_fn=collate_feature_dict, **cfg.get("val_dl_args", {})
    )

    encoder: SeqEncoderContainer = instantiate(cfg["encoder"])
    decoder: AbsDecoder = instantiate(cfg["decoder"])
    module: AbsAE = instantiate(cfg["module"], _recursive_=False)(
        encoder=encoder,
        decoder=decoder,
        amnt_col=amt_column,
        mcc_col=mcc_column,
    )

    if "fast_dev_run" not in cfg["trainer_args"]:
        logger = WandbLogger(project=PROJECT_NAME)
        logger.experiment.config.update(OmegaConf.to_container(cfg))
    else:
        logger = DummyLogger()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        log_every_n_steps=10,
        callbacks=[LearningRateMonitor()],
        **cfg["trainer_args"],
    )

    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
