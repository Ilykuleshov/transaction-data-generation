import os
import logging

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig

from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)

@hydra.main(
    config_path=os.path.join('config', 'dataset'),
    config_name=('ds1'),
    version_base=None
)
def preprocessing(
    cfg: DictConfig
) -> pd.DataFrame:
    
    dir_path: str                   = cfg.dir_path
    ignore_existing_preproc: bool   = cfg.ignore_existing_preproc

    preproc_dir_path = os.path.join(dir_path, 'preprocessed')
    if ignore_existing_preproc:
        logger.info('Preprocessing will ignore all previously saved files')
    logger.info('Preprocessing process started')

    if (
        os.path.exists(os.path.join(preproc_dir_path, 'preproc_dataset.parquet')) and \
        not ignore_existing_preproc
    ):
        return pd.read_parquet(os.path.join(preproc_dir_path, 'preproc_dataset.parquet'))
    


