"""定义损失函数, 优化器, 学习率调度器的接口"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.model import IModel


class ILoss(ABC, Module):
    """损失函数"""

    if TYPE_CHECKING:
        def __call__(self, predict: Tensor, label: Tensor, **kwargs) -> Tensor:
            """计算损失

            Args:
                predict (Tensor): 模型的输出 (批量的序列), shape = (B, ...)
                label (Tensor): 标签, shape = (B, ...)
                length (list[int]): 每条序列的有效长度 ,shape = (B,1)

            Returns:
                Tensor: 批量的平均损失
            """
            ...

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, predict: Tensor, label: Tensor, **kwargs) -> Tensor:
        """计算损失

        Args:
            predict (Tensor): 模型的输出 (批量的序列), shape = (B, ...)
            label (Tensor): 标签, shape = (B, ...)
            length (list[int]): 每条序列的有效长度 ,shape = (B,1)

        Returns:
            Tensor: 损失
        """
        ...


class IOptimizer(ABC):
    """优化器."""

    if TYPE_CHECKING:
        def step(self): ...
        def __init__(self, model: IModel, config: HyperParameterConfig): ...

    # def __init__(self, model: IModel, config: HyperParameterConfig):
    #     self._config = config


    @property
    @abstractmethod
    def config(self): ...

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer: ...


class IScheduler(ABC):
    """学习率调度器"""

    if TYPE_CHECKING:
        def step(self): ...

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self  # type: ignore

    @property
    @abstractmethod
    def config(self): ...
