"""定义损失函数, 优化器, 学习率调度器的接口"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.model import IAgentModel, IModel


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

# region RL functions


class IAgentLoss(ILoss, ABC):
    """强化学习智能体损失函数接口"""

    if TYPE_CHECKING:
        def __call__(self, predicted: Tensor, label: Tensor | None = None, target: Literal["actor", "critic"] | str = "actor") -> Tensor:
            """由于强化学习智能体可能不同的部分需要使用不同的损失函数计算, 因此需要在调用时指定具体的目标"""
            ...

    @abstractmethod
    def forward(self, predicted: Tensor, label: Tensor | None = None, target: Literal["actor", "critic"] | str = "actor") -> Tensor: ...

class IAgentScheduler(IScheduler, ABC):
    """学习率调度器"""
    @property
    @abstractmethod
    def config(self): ...

    @property
    @abstractmethod
    def actor_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        ...

    @property
    @abstractmethod
    def critic_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        ...

    @property
    @abstractmethod
    def critic_other_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        ...

    @property
    @abstractmethod
    def log_alpha_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        ...

class IAgentOptimizer(IOptimizer, ABC):
    """智能体优化器接口.

    必须为 Actor 和 Critic 分别提供优化器
    """

    def __init__(self, model: IModel, config: HyperParameterConfig):
        assert isinstance(model, IAgentModel), "model 必须是 IAgentModel 的实例"
        self._config = config

    @property
    @abstractmethod
    def config(self) -> HyperParameterConfig:
        return self._config

    @property
    @abstractmethod
    def actor_optimizer(self) -> Optimizer:
        """返回 actor 的优化器"""
        raise NotImplementedError

    @property
    @abstractmethod
    def critic_optimizer(self) -> Optimizer:
        """返回 critic 的优化器"""
        raise NotImplementedError

    @property
    @abstractmethod
    def critic_other_optimizer(self) -> Optimizer:
        """返回 其他 critic 优化器"""
        raise NotImplementedError

    @property
    @abstractmethod
    def log_alpha_optimizer(self) -> Optimizer:
        """返回 alpha 的优化器"""
        raise NotImplementedError

# endregion
