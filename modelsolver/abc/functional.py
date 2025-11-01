"""定义损失函数, 优化器, 学习率调度器的接口"""
# region import
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.model import IAgentModel, IModel


# endregion

class ILoss(ABC, Module):
    """损失函数

    + 需要重写 forward 方法.
    + 需要在 `__init__` 中调用 `super().__init__()`
    + 需要重写 TYPE_CHECKING 下的 `__call__` 方法, 以便类型检查工具能够正确识别参数类型
    """

    if TYPE_CHECKING:
        def __call__(self, predict: Tensor, label: Tensor, **kwargs: Tensor) -> Tensor:
            """计算损失

            Args:
                predict (Tensor): 模型的输出 (批量的序列), shape = (B, ...)
                label (Tensor): 标签, shape = (B, ...)
                kwargs (Tensor): 额外的参数, shape = (B, ...)

            Returns:
                损失 (Tensor): 批量的平均损失
            """
            ...

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, predict: Tensor, label: Tensor, **kwargs) -> Tensor:
        """计算损失

        Args:
            predict (Tensor): 模型的输出, shape = (B, ...)
            label (Tensor): 标签, shape = (B, ...)
            kwargs (Tensor): 额外的参数, shape = (B, ...)

        Returns:
            损失 (Tensor): 批量的平均损失
        """
        ...

class IOptimizer(ABC):
    """优化器接口.

    + 需要重写 __getitem__ 方法, 以便根据 key 获取对应的子优化器
    """

    def __init__(self, model: IModel, config: HyperParameterConfig):
        self._config = config


    @property
    def config(self): return self._config

    @abstractmethod
    def __getitem__(self, key: str) -> Optimizer:
        """根据 key 获取对应的优化器

        Args:
            key (str): key

        Returns:
            优化器 (Optimizer): 子优化器. 推荐通过 "all" 键或 "" 返回针对所有参数的优化器
        """
        ...

class IScheduler(ABC):
    """学习率调度器

    + 需要重写 `__getitem__` 方法
    + 需要重写 `config` 属性
    """

    @property
    @abstractmethod
    def config(self): ...

    @abstractmethod
    def __getitem__(self, key: str) -> LRScheduler:
        """根据 key 获取对应的学习率调度器

        Args:
            key (str): key

        Returns:
            学习率调度器 (_LRScheduler): 子学习率调度器. 推荐通过 "all" 键或 "" 返回针对所有参数的学习率调度器
        """
        ...

# region RL functions


class IAgentLoss(ILoss, ABC):
    """强化学习智能体损失函数接口

    + 需要重写 `forward` 方法.
    + 必要时重写 TYPE_CHECKING 下的 `__call__` 方法, 以便类型检查工具能够正确识别参数类型
    """

    if TYPE_CHECKING:
        def __call__(self,
                     predicted: Tensor,
                     label: Tensor | None = None,
                     target: str = "actor",
                     **kwargs: Tensor) -> Tensor:
            """计算损失

            *由于强化学习智能体可能不同的部分需要使用不同的损失函数计算, 因此需要在调用时指定具体的目标*

            Args:
                predicted (Tensor): 预测值
                label (Tensor | None, optional): 标签值. Defaults to None.
                target (str, optional): 目标, 指示计算那部分的损失. Defaults to "actor".
                kwargs (Tensor): 额外的参数.

            Returns:
                损失 (Tensor): 批量的平均损失
            """
            ...

    @abstractmethod
    def forward(self, predicted: Tensor, label: Tensor | None = None, target: str = "actor", **kwargs: Tensor) -> Tensor:
        """计算损失

            *由于强化学习智能体可能不同的部分需要使用不同的损失函数计算, 因此需要在调用时指定具体的目标*

            Args:
                predicted (Tensor): 预测值
                label (Tensor | None, optional): 标签值. Defaults to None.
                target (str, optional): 目标, 指示计算那部分的损失. Defaults to "actor".
                kwargs (Tensor): 额外的参数.

            Returns:
                损失 (Tensor): 批量的平均损失
        """
        ...

class IAgentScheduler(IScheduler, ABC):
    """学习率调度器

    + 需要重写 __getitem__ 方法, 以便根据 key 获取对应的子学习率调度器
        + key: "actor", "critic", "critic_other", "log_alpha"
    + 需要重写 `config` 属性
    """

    @abstractmethod
    def __getitem__(self, key: str) -> LRScheduler:
        ...

    @property
    def actor_scheduler(self) -> LRScheduler: return self["actor"]

    @property
    def critic_scheduler(self) -> LRScheduler: return self["critic"]

    @property
    def critic_other_scheduler(self) -> LRScheduler: return self["critic_other"]

    @property
    def log_alpha_scheduler(self) -> LRScheduler: return self["log_alpha"]

class IAgentOptimizer(IOptimizer, ABC):
    """智能体优化器接口.

    + 需要重写 __getitem__ 方法, 以便根据 key 获取对应的子优化器
        + key: "actor", "critic", "critic_other", "log_alpha"

    """

    def __init__(self, model: IModel, config: HyperParameterConfig):
        assert isinstance(model, IAgentModel), "model 必须是 IAgentModel 的实例"
        super().__init__(model, config)

    @property
    def actor_optimizer(self) -> Optimizer: return self["actor"]

    @property
    def critic_optimizer(self) -> Optimizer: return self["critic"]

    @property
    def critic_other_optimizer(self) -> Optimizer: return self["critic_other"]

    @property
    def log_alpha_optimizer(self) -> Optimizer: return self["log_alpha"]

# endregion
