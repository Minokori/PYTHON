# regiopn imports
import logging
from collections import deque
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from gymnasium.envs.classic_control import Continuous_MountainCarEnv
from torch import Tensor, from_numpy, tensor

from modelsolver.abc.environment import IEnvironment


logging.basicConfig(level=logging.DEBUG, filename="mountaincar.log", filemode="w",encoding="utf-8")
np.set_printoptions(precision=2, suppress=True)
# endregion

@dataclass
class MountainCarConfig:
    """山地车环境配置"""
    terminated_delta: int = -1
    """是否启用终止状态检测. 设置为 >0 的值时, 当连续若干时间步达到数值状态时, 环境将进入终止状态."""
    truncated_time: int = -1
    """时间步截断. 设置为 <0 则不启用截断."""


class MountainCarEnvironment(Continuous_MountainCarEnv,IEnvironment):
    """山地车环境"""

    def __init__(self, config: MountainCarConfig) -> None:
        super().__init__(render_mode="human")
        open("mountaincar.log", "w").close()  # 清空日志文件
        self.history_buffer = deque(maxlen=config.terminated_delta) if config.terminated_delta > 0 else None
        """历史状态缓冲区. 不启用终止状态时, 为 None"""
        self.truncated_time = config.truncated_time if config.truncated_time > 0 else None
        """时间步截断时间. 不启用截断时, 为 None"""
        self.time = 0 if config.truncated_time > 0 else None
        """当前仿真时间. 不启用截断时, 为 None"""

    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        self.time = 0 if self.time is not None else None
        ob, info = super().reset()
        return from_numpy(ob).float().reshape(-1), tensor(0).float().reshape(1), tensor(0).float().reshape(1), False, info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, reward, terminated, truncated, info = super().step(action.cpu().detach().numpy())

        state = from_numpy(ob).float().reshape(-1)
        self.history_buffer.append(state) if self.history_buffer is not None else None
        if self.time is not None:
            self.time += 1

        return state, tensor(reward).float().reshape(1), self.is_terminated(), self.is_truncated(), info

    def is_terminated(self) -> Tensor:
        if self.history_buffer is None:
            return tensor(0).float().reshape(1)
        for history_state in self.history_buffer:
            if not self.is_success(history_state):
                return tensor(0).float().reshape(1)
        return tensor(1).float().reshape(1)

    def is_truncated(self) -> bool:
        if self.truncated_time is None or self.time is None:
            return False
        elif self.time >= self.truncated_time:
            return True
        else:
            return False

    def is_success(self, state: Tensor) -> bool:
        return torch.sum(torch.abs(state - tensor([self.goal_position, self.goal_velocity]))).item() < 1e-3

    def build_environment(self) -> Self:
        return self
