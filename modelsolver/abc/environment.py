"""环境模块, 定义了环境接口, 包括强化学习环境接口"""
from abc import ABC, abstractmethod
from collections import deque
from typing import Self

from torch import Tensor

from modelsolver.abc.config import EnvironmentConfig


# TODO step 返回的state包含goal, 方便HER算法的实现
# TODO 抽象出一个IReward接口.


class IEnvironment(ABC):
    """包装环境的接口.

    + 需要重写环境的 `reset` 和 `step` 方法, 使其返回相同的数据结构, 并且数据类型为 `Tensor.cpu()`
    + 需要重写环境的 `step` 方法, 使其接受类型为 `Tensor` 的动作
    + 需要 `build_environment` 方法, 确保环境被构建
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

    # TODO 删除, 减少耦合
    @abstractmethod
    def build_environment(self, **kwargs) -> Self:
        """调用一次以确保环境被构建"""
        ...
        ...

    @property
    def ZERO_ACTION(self) -> Tensor:
        """环境的零动作, 用于经验回放池预热等场景. 预期得到 Tensor.cpu()"""
        ...

    @property
    def GOAL(self) -> Tensor:
        """环境的目标状态, 用于HER等算法. 预期得到 Tensor.cpu()"""
        ...


