from torch import nn, Tensor
import typing as tp
from src.networks.decoders.base import AbsDecoder


class LSTMDecoder(nn.LSTM, AbsDecoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int = 0,
        num_layers: int = 1,
        bidir: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            proj_size=proj_size,
            batch_first=True,
        )

        self.output_size = hidden_size * num_layers * (bidir + 1)

    def forward(self, x: Tensor, hx: tp.Optional[tp.Tuple[Tensor, Tensor]] = None):
        return super().forward(x, hx)[0]
