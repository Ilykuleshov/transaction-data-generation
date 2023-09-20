import os
import pickle
import typing

import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.utils.logging_utils import get_logger
from src.utils.data_utils import split_into_samples
from src.anomaly_scheme import by_mcc_percentiles


logger = get_logger(name=__name__)

def preprocessing(
    cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame]: 
    
    dir_path: str                   = cfg['dir_path']
    ignore_existing_preproc: bool   = cfg['ignore_existing_preproc']
    # drop_currency: bool             = cfg['preproc']['drop_currency']
    time_delta: int                 = cfg['preproc']['time_delta']
    len_min: int                    = cfg['preproc']['len_min']
    len_max: int                    = cfg['preproc']['len_max']
    anomaly_strategy: str           = cfg['anomaly_strategy']
    percentile: float               = cfg['percentile']

    user_column: str = cfg["user_column"]
    mcc_column: str = cfg["mcc_column"]
    transaction_amt_column: str = cfg["transaction_amt_column"]
    transaction_dttm_column: str = cfg["transaction_dttm_column"]

    preproc_dir_path = os.path.join(dir_path, 'preprocessed')
    if not os.path.exists(preproc_dir_path):
        logger.warning('Preprocessing folder does not exist. Creating...')
        os.mkdir(preproc_dir_path)
    if ignore_existing_preproc:
        logger.info('Preprocessing will ignore all previously saved files')

    if not ignore_existing_preproc and os.path.exists(os.path.join(preproc_dir_path, "preproc_dataset.parquet")):
        data_srt = pd.read_parquet(os.path.join(preproc_dir_path, 'preproc_dataset.parquet'))
    # TODO: mebe fix preprocessing
    else:
        df_orig = pd.read_parquet(os.path.join(dir_path, 'data.parquet'))

        # logger.info('Transfering timestamp to the datetime format')
        # df_orig['transaction_dttm'] = pd.to_datetime(
        #     df_orig['transaction_dttm'],
        #     format='%Y-%m-%d %H:%M:%S'
        # )
        # logger.info('Done!')

        # df_orig.drop(
        #     index=df_orig[df_orig['mcc_code'] == -1].index,
        #     axis=0,
        #     inplace=True
        # )

        # if drop_currency:
        #     logger.info('Dropping currency_rk')
        #     df_orig.drop(
        #         index=df_orig[df_orig['currency_rk'] != 48].index,
        #         axis=0,
        #         inplace=True
        #     )
        #     df_orig.drop(columns=['currency_rk'], axis=1, inplace=True)
        #     logger.info('Done!')

        # if (
        #     os.path.exists(os.path.join(preproc_dir_path, 'mcc2id.dict')) and \
        #     not ignore_existing_preproc
        # ):
        #     with open(os.path.join(preproc_dir_path, 'mcc2id.dict'), 'rb') as f:
        #         mcc2id = dict(pickle.load(f))
        # else:
        #     mcc2id = dict(zip(
        #         df_orig['mcc_code'].unique(), 
        #         np.arange(df_orig['mcc_code'].nunique()) + 1
        #     ))
        #     with open(os.path.join(preproc_dir_path, 'mcc2id.dict'), 'wb') as f:
        #         pickle.dump(mcc2id, f)
        
        # df_orig['mcc_code'] = df_orig['mcc_code'].map(mcc2id)

        # df_orig['is_income'] = (df_orig['transaction_amt'] > 0).astype(np.int32)
        # df_orig['transaction_amt'] = df_orig[['transaction_amt', 'is_income']].apply(
        #     lambda t: np.log(t[0]) if t[1] else np.log(-t[0]),
        #     axis=1
        # )

        # if (
        #     os.path.exists(os.path.join(preproc_dir_path, 'user2id.dict')) and \
        #     not ignore_existing_preproc
        # ):
        #     with open(os.path.join(preproc_dir_path, 'user2id.dict'), 'rb') as f:
        #         user2id = dict(pickle.load(f))
        # else:
        #     user2id = dict(zip(
        #         df_orig['user_id'].unique(), 
        #         np.arange(df_orig['user_id'].nunique()) + 1
        #     ))
        #     with open(os.path.join(preproc_dir_path, 'user2id.dict'), 'wb') as f:
        #         pickle.dump(user2id, f)
        # df_orig['user_id'] = df_orig['user_id'].map(user2id)
        
        data_srt = df_orig.sort_values(['user_id','timestamp']).reset_index(drop=True)

        logger.info('Start splitting into samples')
        split_into_samples(data_srt, time_delta, len_min, len_max, user_column_name=user_column, data_column_name=transaction_dttm_column)
        logger.info('Done!')
        # logger.info('Anomaly splitting')
        # if anomaly_strategy == 'quantile':
        #     by_mcc_percentiles(data_srt, percentile=percentile, mcc_column=mcc_column, transaction_amt_column=transaction_amt_column, binary_income_column=None)
        # logger.info('Done!')
        # anomaly_samples = data_srt['target'].sum()
        # normal_samples = data_srt.shape[0] - anomaly_samples
        # logger.info(f'Normal samples count - {int(normal_samples)}. Anomaly Samples - {int(anomaly_samples)}')

        data_srt.to_parquet(os.path.join(preproc_dir_path, 'preproc_dataset.parquet'))

    if (
        os.path.exists(os.path.join(preproc_dir_path, 'agg_dataset.parquet')) and \
        not ignore_existing_preproc
    ):
        data_agg = pd.read_parquet(os.path.join(preproc_dir_path, 'agg_dataset.parquet'))
    
    else:
        data_agg = data_srt.groupby('sample_label').agg({
            user_column: lambda x: x.iloc[0],
            mcc_column: lambda x: x.tolist(),
            transaction_amt_column: lambda x: x.tolist(),
        })
        data_agg.to_parquet(os.path.join(preproc_dir_path, 'agg_dataset.parquet'))

    return data_agg, data_srt
