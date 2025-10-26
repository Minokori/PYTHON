from abc import ABC, abstractmethod
from typing import Any, Self

from numpy import float32
from numpy.typing import NDArray
from torch import Tensor


class IEnvironment(ABC):
    """包装环境的接口.

    + 需要重写环境的 `reset` 和 `step` 方法, 使其返回相同的数据结构, 并且数据类型为 Tensor.cuda()
    + 需要重写环境的 `step` 方法, 使其接受 Tensor.cuda()
    + Tensor.cuda() 需要在环境内部进行处理, 以确保与外部接口的一致性
    """
    @abstractmethod
    def step(self, action: NDArray[float32] | Tensor) -> tuple[NDArray[float32], float, bool, bool, dict]: ...

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[NDArray[float32], float, bool, bool, dict]:
        """和 step 保持一致, 返回 (observation, reward, terminated, truncated, info).

        *对于 reset() 而言, 仅 observation 有意义*
        """
        ...

    @abstractmethod
    def build_environment(self, config: Any | None = None, render_mode: str = "human", ) -> Self: ...
