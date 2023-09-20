import os
from pathlib import Path
import pickle
import typing as tp

import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.utils.logging_utils import get_logger
from src.utils.data_utils import split_into_samples
from src.anomaly_scheme import by_mcc_percentiles
from ptls.preprocessing import PandasDataPreprocessor


logger = get_logger(name=__name__)


def preprocessing(cfg: DictConfig) -> tp.List[tp.Dict]:
    dir_path: Path = Path(cfg["dir_path"])
    ignore_existing_preproc: bool = cfg["ignore_existing_preproc"]
    amount_quantile: float = cfg["amount_quantile"]

    mcc_column: str = cfg["mcc_column"]
    amt_column: str = cfg["amt_column"]

    if ignore_existing_preproc:
        logger.info("Preprocessing will ignore all previously saved files")

    preproc_df_path = dir_path / "preproc_df.parquet"
    if not ignore_existing_preproc and preproc_df_path.exists():
        df = pd.read_parquet(preproc_df_path)
    else:
        df = pd.read_csv(dir_path / "raw_data.csv")
        logger.info("Transfering timestamp to the datetime format")
        df["TRDATETIME"] = pd.to_datetime(
            df["TRDATETIME"], format=r"%y%b%d:%H:%M:%S"
        )
        logger.info("Done!")

        currency_mode = df["currency"].mode().item()
        logger.info(f"Dropping transactions with currency != {currency_mode=}")
        df = df[df["currency"] == currency_mode]
        logger.info("Done!")

        amount_quantile_val = df["amount"].quantile(amount_quantile)
        logger.info(f"Dropping transactions larger than {amount_quantile_val=}...")
        df = df[df["amount"] < amount_quantile_val]
        logger.info("Done!")

        df.drop(
            columns=["channel_type", "PERIOD", "currency", "trx_category", "target_flag", "target_sum"],
            inplace=True
        )

        df.rename(columns={"amount": amt_column, "MCC": mcc_column}, inplace=True)
        df["amount"] = df["amount"].astype("float32")
        df.to_parquet(preproc_df_path)

    preprocessor_path = dir_path / "preprocessor.p"
    if not preprocessor_path.exists() or ignore_existing_preproc:
        preprocessor = PandasDataPreprocessor(
            "cl_id",
            "TRDATETIME",
            cols_category=[mcc_column],
            cols_numerical=[amt_column],
            return_records=True,
        )  # type: ignore

        logger.info("Fitting Pandas preprocessor")
        dataset: tp.List[tp.Dict] = preprocessor.fit_transform(df)  # type: ignore
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
    else:
        with open(preprocessor_path, "rb") as f:
            dataset = pickle.load(f).transform(df)

    return dataset
