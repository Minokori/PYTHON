from abc import ABC, abstractmethod
from typing import Self

from torch import Tensor


class IEnvironment(ABC):
    """包装环境的接口.

    + 需要重写环境的 `reset` 和 `step` 方法, 使其返回相同的数据结构, 并且数据类型为 Tensor.cpu()
    + 需要重写环境的 `step` 方法, 使其接受 Tensor
    + Tensor.device() 需要在环境内部进行处理, 以确保与外部接口的一致性
    """
    @abstractmethod
    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:
        """执行一步环境交互, 返回 (observation, reward, terminated, truncated, info), tensor 在 cpu 上
        """
        ...

    @abstractmethod
    def reset(self) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """和 step 保持一致, 返回 (observation, reward, terminated, truncated, info).

        *对于 reset() 而言, 仅 observation 有意义*
        """
        ...

    @abstractmethod
    def build_environment(self, **kwargs) -> Self:
        """调用一次以确保环境被构建"""
        ...
