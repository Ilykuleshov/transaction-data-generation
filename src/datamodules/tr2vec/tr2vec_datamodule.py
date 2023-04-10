import typing

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

import pandas as pd

import pytorch_lightning as pl

from omegaconf import DictConfig

from .tr2vec_dataset import T2VDataset
from utils.logging_utils import get_logger

logger = get_logger(name=__name__)


class T2VDatamodule(pl.LightningDataModule):

    def __init__(self, cfg: DictConfig, sequences: pd.Series) -> None:
        super().__init__()
        self.window_size: int = cfg['window_size']
        self.batch_size: int = cfg['learning_params']['batch_size']

        logger.info('Making raw sequences')
        sequences = [torch.LongTensor(seq) for seq in sequences]

        test_fraq: float = cfg['data_split']['test_fraq']
        train_val_fraq = 1 - test_fraq
        val_fraq: float = cfg['data_split']['val_fraq'] * train_val_fraq
        train_fraq = 1 - val_fraq - test_fraq

        logger.info('Creating full dataset')
        full_dataset = T2VDataset(sequences, self.window_size)

        logger.info('Making subsets')
        self.train, self.val, self.test = random_split(full_dataset, (train_fraq, val_fraq, test_fraq))
        logger.info('Done')

    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            self.batch_size,
            True,
            collate_fn=self.tr2vec_collate
        )


    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            self.batch_size,
            collate_fn=self.tr2vec_collate
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            self.batch_size,
            collate_fn=self.tr2vec_collate
        )
        

    @staticmethod
    def tr2vec_collate(batch: typing.List[torch.Tensor]):
        ctx_mccs, center_mccs, ctx_lengths = zip(*batch)
        ctx_mccs = pad_sequence(ctx_mccs, batch_first=True, padding_value=0)
        ctx_lengths = torch.LongTensor(ctx_lengths)
        center_mccs = torch.LongTensor(center_mccs)
        return ctx_mccs, center_mccs, ctx_lengths
