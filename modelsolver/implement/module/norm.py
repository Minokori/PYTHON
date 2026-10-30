"""预实现的 norm 模块"""
# pylint: disable=W2301
from typing import TYPE_CHECKING

from torch import Tensor, ones, sqrt
from torch.nn import Parameter

from modelsolver.abc.model import IModule


class RmsNorm(IModule):
    """均方根归一化（RMSNorm）模块

    x / sqrt(mean(x²)+eps) * weight
    """
    if TYPE_CHECKING:
        def __call__(self, x: Tensor) -> Tensor:
            """均方根归一化后的输出，形状与输入一致"""
            ...
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = Parameter(ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """前向传播

        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, hidden_size)

        Returns:
            归一化后的张量，形状与输入一致
        """
        #
        # $\hat{x} = x / \sqrt{\text{mean}(x^2)+\epsilon}$
        #
        rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

