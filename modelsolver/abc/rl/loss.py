from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from torch import Tensor

from modelsolver.abc.functional import ILoss


class IAgentLoss(ILoss, ABC):
    """强化学习智能体损失函数接口"""

    if TYPE_CHECKING:
        def __call__(self, predicted: Tensor, label: Tensor | None = None, target: Literal["actor", "critic"] | str = "actor") -> Tensor:
            """由于强化学习智能体可能不同的部分需要使用不同的损失函数计算, 因此需要在调用时指定具体的目标"""
            ...

    @abstractmethod
    def forward(self, predicted: Tensor, label: Tensor | None = None, target: Literal["actor", "critic"] | str = "actor") -> Tensor: ...
