import typing

import pandas as pd

from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from src.utils.logging_utils import get_logger
from .dataset import AEDataset

logger = get_logger(name=__name__)


class AEDataModule(LightningDataModule):

    def __init__(
        self,
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        user_column: str,
        mcc_code_column: str,
        transaction_amt_column: str,
        train_val_ratio: float,
        batch_size:int,
        max_len: int,
        num_workers: int,
        anomaly_fraq: float,
        only_normal: bool,
        binarize: bool,
        *args, **kwargs
    ) -> None:
        logger.info('Creating DataLoader')
        super().__init__()
        self.train_val_frame        = data_train
        self.test_frame             = data_test
        self.user_column            = user_column
        self.mcc_code_column        = mcc_code_column
        self.transaction_amt_column = transaction_amt_column
        self.train_val_ratio        = train_val_ratio
        self.max_len                = max_len
        self.num_workers            = num_workers
        self.batch_size             = batch_size
        self.anomaly_fraq           = anomaly_fraq
        self.only_normal            = only_normal
        self.binarize               = binarize

        self.train: AEDataset       = None
        self.val: AEDataset         = None
        self.test: AEDataset        = None

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            train_val = AEDataset(
                self.train_val_frame,
                self.user_column,
                self.mcc_code_column,
                self.transaction_amt_column,
                self.anomaly_fraq,
                self.only_normal,
                self.binarize
            )
            length = len(train_val)
            self.train, self.val = random_split(
                train_val,
                (
                    length - int(length * self.train_val_ratio),
                    int(length * self.train_val_ratio)
                )
            )
        elif stage == 'test':
            self.test = AEDataset(
                self.test_frame,
                self.user_column,
                self.mcc_code_column,
                self.transaction_amt_column,
                self.anomaly_fraq,
                self.only_normal,
                self.binarize
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )

    @staticmethod
    def _collate_fn(batch: typing.List[typing.Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.DoubleTensor,
        int,
        float
    ]]):
        user_id, mcc_codes, is_income, transaction_amt, lengths, targets = zip(*batch)
        user_id = torch.LongTensor(user_id)
        mcc_codes = pad_sequence(mcc_codes, batch_first=True, padding_value=0).long()
        is_income = pad_sequence(is_income, batch_first=True, padding_value=0).float()
        transaction_amt = pad_sequence(transaction_amt, batch_first=True, padding_value=0).float()
        lengths = torch.LongTensor(lengths)
        targets = torch.Tensor(targets)
        return user_id, mcc_codes, is_income, transaction_amt, lengths, targets
