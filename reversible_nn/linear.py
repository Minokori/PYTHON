import torch
from torch import nn

from modelsolver.abc.model import IModule


class Linear(IModule):
    """可逆的全连接层
    """

    def __init__(self, channels_in, channels_out):
        super().__init__()
        assert channels_in <= channels_out, "输入维度应小于等于输出维度"
        self._layer: torch.nn.Linear = nn.Linear(channels_out, channels_out)
        self._shape = (channels_in, channels_out)
        """映射的网络层"""
        torch.nn.init.xavier_uniform_(self._layer.weight)
        torch.nn.init.zeros_(self._layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device, dtype = x.device, x.dtype

        x = torch.cat([x, torch.zeros(B, T, self._shape[1] - self._shape[0], device=device, dtype=dtype)], dim=2)

        return self._layer(x).type(dtype).to(device)

    def reverse(self, seqLike_input_batch: torch.Tensor) -> torch.Tensor:
        W, B = self._get_reverse_weight_and_bias()

        z = seqLike_input_batch @ W + B
        return z[:, :, : self._shape[0]]

    def _get_reverse_weight_and_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """初始化逆映射网络

        Returns:
            nn.Module: 逆映射网络
        """

        W, B = self._layer.weight.data, self._layer.bias.data

        _W = W.inverse()
        if not torch.isfinite(_W).all():
            _W = torch.linalg.pinv(W).T
        else:
            _W = _W.T

        _B = -_W @ B

        return _W, _B
        # return W2, B2
