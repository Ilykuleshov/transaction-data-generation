from datetime import timedelta

import numpy as np
import pandas as pd

from tqdm.auto import tqdm


def split_into_samples(
    data: pd.DataFrame,
    time_delta: int = 7,
    len_min: int = 40,
    len_max: int = 120,
    user_column_name: str = 'user_id',
    data_column_name: str = 'transaction_dttm'
) -> None:
    
    label_column = np.zeros(data.shape[0])
    start_user = data.iloc[0][user_column_name]

    start_user = data.iloc[0]['user_id']
    index = 0
    start_time = data.iloc[0]['transaction_dttm']
    count = 0

    for i in tqdm(range(len(data))):
        curr_time = data.iloc[i][data_column_name]
        curr_user = data.iloc[i][user_column_name]
        count += 1
        if (
            (curr_time > start_time + timedelta(days=time_delta) and count > 40) or \
            curr_user != start_user
        ):
            count = 1
            index += 1
            start_time = curr_time
            start_user = curr_user
        label_column[i] = index
    
    data['sample_label'] = label_column
    data['count_temp'] = data.groupby('sample_label')['sample_label'].count()
    data.drop(
        index=data[(data['count_temp'] < len_min) | (data['count_temp'] > len_max)].index,
        axis=0,
        inplace=True
    )
    data.drop(columns='count_temp', axis=1, inplace=True)
