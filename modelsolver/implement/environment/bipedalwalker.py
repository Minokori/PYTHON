# regiopn imports
import logging
from collections import deque
from dataclasses import dataclass
from typing import Self

import numpy as np
from gymnasium.envs.box2d import BipedalWalker
from torch import Tensor, from_numpy, tensor

from modelsolver.abc.environment import IEnvironment


logging.basicConfig(level=logging.DEBUG, filename="bipedalwalker.log", filemode="w",encoding="utf-8")
np.set_printoptions(precision=2, suppress=True)
# endregion

@dataclass
class BipedalWalkerConfig:
    """漫步环境配置"""
    terminated_delta: int = -1
    """是否启用终止状态检测. 设置为 >0 的值时, 当连续若干时间步达到数值状态时, 环境将进入终止状态."""
    truncated_time: int = -1
    """时间步截断. 设置为 <0 则不启用截断."""


class BipedalWalkerEnvironment(IEnvironment):
    """漫步环境"""

    def __init__(self) -> None:
        self._env = BipedalWalker(render_mode="human")

    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, info = self._env.reset()
        return from_numpy(ob).float().reshape(-1), tensor(0).float().reshape(1), tensor(0).float().reshape(1), False, info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, reward, terminated, truncated, info = self._env.step(action.cpu().detach().numpy())
        state = from_numpy(ob).float().reshape(-1)
        return state, tensor(reward).float().reshape(1), tensor(terminated).float().reshape(1), truncated, info

    def build_environment(self) -> Self:
        return self
