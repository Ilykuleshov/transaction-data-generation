import typing

import torch

from .lstm import LSTMAE


class MaskedLSTMAE(LSTMAE):

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        mcc_embed_dim: int,
        n_vocab_size: int,
        loss_weights: tuple[float, float, float],
        freeze_embed: bool,
        unfreeze_after: int,
        lr: float,
        weight_decay: float,
        use_user_embedding: bool,
        user_embedding_size: int,
        rand_mask_rate: float,
        *args: typing.Any, **kwargs: typing.Any
    ) -> None:
        super().__init__(
            embed_dim, 
            num_layers, 
            mcc_embed_dim, 
            n_vocab_size, 
            loss_weights, 
            freeze_embed, 
            unfreeze_after, 
            lr, 
            weight_decay, 
            use_user_embedding,
            user_embedding_size,
            *args, **kwargs
        )
        self.save_hyperparameters({
            'rand_mask_rate': rand_mask_rate
        })

    def _rand_mask(lengths: torch.Tensor) -> torch.Tensor:
        
