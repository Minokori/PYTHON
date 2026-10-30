"""对 torch.nn.Linear 的简单包装. 扩展了一些初始化参数的方法."""

from typing import Any, Literal, Self

from git import TYPE_CHECKING
from torch import Tensor, nn

from modelsolver.abc.model import IModule


Method = Literal["default", "xavier_uniform","zeros"]

class Linear(IModule, nn.Linear):
    """对 torch.nn.Linear 的简单包装. 扩展了一些初始化参数的方法."""

    if TYPE_CHECKING:
        def __call__(self, x: Tensor) -> Tensor:
            """线性连接层

            Args:
                x (Tensor): 输入

            Returns:
                输出 (Tensor): 将输入的最后一个维度变为 `out_features`
            """
            ...

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device:Any = None,
        dtype: Any = None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

    def init_weights(self, method: Method|float = "default") -> Self:
        """初始化权重.

        Args:
            method (Method|float, optional): 初始化方法.
                - "default": PyTorch nn.Linear 的默认初始化：kaiming_uniform(a=√5)
                - "xavier_uniform": Xavier 均匀分布初始化
                - "zeros": 全零初始化
                - float: 正态分布初始化，均值为 0，标准差为 method
        """
        match method:
            case "default":
                # PyTorch nn.Linear 的默认初始化：kaiming_uniform(a=√5)。
                nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            case "xavier_uniform":
                nn.init.xavier_uniform_(self.weight)
            case "zeros":
                nn.init.zeros_(self.weight)
            case float():
                nn.init.normal_(self.weight, std=method)
            case _:
                raise ValueError(f"非法的初始化方法: {method}")
        return self

    def init_bias(self, method: Method|float = "default") -> Self:
        """初始化偏置.

        Args:
            method (Method|float, optional): 初始化方法.
                - "default": PyTorch nn.Linear 的默认初始化：kaiming_uniform(a=√5)
                - "xavier_uniform": Xavier 均匀分布初始化
                - "zeros": 全零初始化
                - float: 正态分布初始化，均值为 0，标准差为 method
        """
        match method:
            case "default":
                # PyTorch nn.Linear 的默认初始化：kaiming_uniform(a=√5)。
                nn.init.kaiming_uniform_(self.bias, a=5 ** 0.5)
            case "xavier_uniform":
                nn.init.xavier_uniform_(self.bias)
            case "zeros":
                nn.init.zeros_(self.bias)
            case float():
                nn.init.normal_(self.bias, std=method)
            case _:
                raise ValueError(f"非法的初始化方法: {method}")
        return self