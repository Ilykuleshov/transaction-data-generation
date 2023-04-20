import typing

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)


class AEDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        user_column: str = 'user_id',
        mcc_code_column: str = 'mcc_code',
        transaction_amt_column: str = 'transaction_amt',
        anomaly_fraq: float = .2,
        only_normal: bool = True,
        binarize: bool = False
    ) -> None:
        super().__init__()
        # Drop anomaly if required
        if only_normal:
            shape_before = data.shape[0]
            data.drop(index=data[data['target'] > anomaly_fraq].index, inplace=True)
            logger.info(f'Samples droped - {shape_before - data.shape[0]}')

        self.user_ids = [
            torch.tensor(user_id, dtype=torch.int32) for user_id in data[user_column]
        ] 
        self.mcc_codes = [
            torch.tensor(mcc_code, dtype=torch.int32) for mcc_code in data[mcc_code_column]
        ]
        self.is_income = [
            torch.tensor(income, dtype=torch.int32) for income in data['is_income']
        ]
        self.transaction_amt = [
            torch.tensor(amt) for amt in data[transaction_amt_column]
        ]
        self.targets = data['target'].values
        if binarize:
            self.targets = (self.targets > anomaly_fraq).astype(np.int32)

    def __getitem__(self, index) -> typing.Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.DoubleTensor,
        float,
        int
    ]:
        user_id = self.user_ids[index]
        mcc_codes = self.mcc_codes[index]
        is_income = self.is_income[index]
        transaction_amt = self.transaction_amt[index]
        target = self.targets[index]
        length = len(mcc_codes)

        return user_id, mcc_codes, is_income, transaction_amt, length, target
    

    def __len__(self) -> int:
        return self.targets.shape[0]
