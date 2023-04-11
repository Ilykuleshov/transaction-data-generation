import typing

import torch
from torch.utils.data import Dataset

from tqdm.auto import tqdm


class T2VDataset(Dataset):
    
    def __init__(
        self,
        mcc_sequences: typing.List[torch.LongTensor],
        window_size: int
    ) -> None:
        super().__init__()

        self.id2seq_id = []
        self.id2offset = [] 

        self.window_size = window_size
        mcc_sequences = [seq for seq in mcc_sequences if len(seq) > 1]
        lens = [len(seq) for seq in mcc_sequences]

        for seq_id, l in tqdm(enumerate(lens)):
            self.id2seq_id += [seq_id] * l
            self.id2offset += list(range(l))
        
        self.mcc_seqs = mcc_sequences
    

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_id, offset = self.id2seq_id[index], self.id2offset[index]
        mcc_seq = self.mcc_seqs[seq_id]
        center_mcc = mcc_seq[offset]
        left, right = max(offset - self.window_size, 0), min(offset + self.window_size, len(mcc_seq))
        ctx_mcc = torch.cat([mcc_seq[left:offset], mcc_seq[offset + 1:right]])
        ctx_length = len(ctx_mcc)
        return ctx_mcc, center_mcc, ctx_length


    def __len__(self) -> int:
        return len(self.id2seq_id)
