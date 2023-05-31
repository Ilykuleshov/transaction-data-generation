from datetime import timedelta
import typing

import numpy as np
import pandas as pd

import torch

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

    start_user = data.iloc[0][user_column_name]
    index = 0
    start_time = data.iloc[0][data_column_name]
    count = 0
    index_to_drop = []
    increment = 0

    for i in tqdm(range(len(data)), leave=True):
        curr_time = data.iloc[i][data_column_name]
        curr_user = data.iloc[i][user_column_name]
        count += 1
        if curr_user != start_user:
            if len_min > count < len_max:
                index_to_drop.append(index)
            count = 1
            index += 1
            start_user = curr_user
            start_time = curr_time
        elif curr_time > start_time + timedelta(days=time_delta):
            start_time = curr_time
            if count >= len_min:
                if len_min > count < len_max:
                    index_to_drop.append(index)
                count = 1
                index += 1
        label_column[i] = index
    if len_min > count < len_max:
        index_to_drop.append(index)
    
    data['sample_label'] = np.array(label_column)
    data.drop(index=data[data['sample_label'].isin(index_to_drop)].index, axis=0, inplace=True)
    data['sample_label'] = data['sample_label'].astype(np.int32)


def compute_mask(lengths: torch.Tensor, device: str) -> torch.Tensor:
    max_length = lengths.max().detach().cpu().item()
    mask = torch.arange(max_length) \
            .expand(len(lengths), max_length) \
            .to(device) < lengths.unsqueeze(1)
    mask.requires_grad_(False).unsqueeze_(-1)
    return mask


def compute_rand_mask(
    B: int, L: int, H: int, rand_rate: float, device: str
) -> torch.Tensor:
    mask = (torch.rand((B, L)) < rand_rate)
    mask = mask.unsqueeze(-1).repeat(1, 1, H).to(device)
    mask.requires_grad_(False)
    return mask


# Method for returning original padding
def trim_out_seq(
    seqs_to_trim: typing.Union[tuple[torch.Tensor, ...], torch.Tensor],
    mask: torch.Tensor
) -> list[torch.Tensor]:
    if isinstance(seqs_to_trim, torch.Tensor):
        trimmed_seqs = seqs_to_trim.masked_fill(~mask, 0)
    else:
        trimmed_seqs = list(map(lambda seq: seq.masked_fill(~mask, 0), seqs_to_trim))

    return trimmed_seqs
