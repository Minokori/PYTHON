"""多层感知机的默认实现"""
# pylint: disable=W2301
from typing import TYPE_CHECKING, Self

import torch
from torch.nn import GELU, LayerNorm, Module

from modelsolver.abc.model import IModule
from modelsolver.implement.module.linear import Linear, Method


class BaseMLP(IModule):
    """多层感知机的基类.

    该类的基本结构如下:
    - Linear (输入维度 -> 4*输入维度)
    - 激活函数 (默认为 GELU. 若需要自定义, 则需要重写 `_init_activation` 方法)
    - Linear (4*输入维度 -> 输入维度)

    """

    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """输出形状与输入形状一致."""
            ...

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear1: Linear = Linear(channels, 4 * channels)
        """第一个线性层"""
        self.linear2: Linear = Linear(4 * channels, channels)
        """第二个线性层"""
        self.activation = self._init_activation()
        """激活函数"""

    def init_weights(self, weight_init:tuple[Method|float, Method|float]) -> Self:
        """初始化权重

        Args:
            weight_init (tuple[Method | float, Method | float]): 权重初始化方法或值的元组
        """
        self.linear1.init_weights(weight_init[0])
        self.linear2.init_weights(weight_init[1])
        return self

    def init_bias(self, bias_init:tuple[Method|float, Method|float]) -> Self:
        """初始化偏置

        Args:
            bias_init (tuple[Method | float, Method | float]): 偏置初始化方法或值的元组
        """
        self.linear1.init_bias(bias_init[0])
        self.linear2.init_bias(bias_init[1])
        return self

    def _init_activation(self)->Module:
        """初始化激活函数

        Returns:
            Module: 激活函数模块
        """
        return GELU()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x



class LayerNormMLP(BaseMLP):
    """带有 LayerNorm 的多层感知机.

    input: (B, T, C) -> output: (B, T, C)


    LayerNorm + (B,T, C)->(B,T,4C) + GELU + (B,T,4C)->(B,T,C)

    每个线性层的权重使用 xavier_uniform 初始化，偏置初始化为零

    """
    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """输出形状与输入形状一致"""
            ...

    def __init__(self, channels: int):
        """初始化方法

        Args:
            channels (int): 输入/输出特征维度
        """
        super().__init__(channels)
        self.layer_norm = LayerNorm(channels, elementwise_affine=False)
        self.init_weights(("xavier_uniform", "xavier_uniform")).init_bias(("zeros", "zeros"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = super().forward(x)
        return x
