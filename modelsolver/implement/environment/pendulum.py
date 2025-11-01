# regiopn imports
from collections import deque
from dataclasses import dataclass
from typing import Self

import torch
from gymnasium.envs.classic_control import PendulumEnv
from torch import Tensor, from_numpy, tensor

from modelsolver.abc.environment import IEnvironment


# endregion

@dataclass
class PendulumConfig:
    """摆锤环境配置"""
    terminated_delta: int = -1
    """是否启用终止状态检测. 设置为 >0 的值时, 当连续若干时间步达到数值状态时, 环境将进入终止状态."""
    truncated_time: int = -1
    """时间步截断. 设置为 <0 则不启用截断."""


class PendulumEnvironment(PendulumEnv, IEnvironment):
    """摆锤环境"""

    def __init__(self, config: PendulumConfig = PendulumConfig()) -> None:
        super().__init__(render_mode="human")
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
        ob, reward, terminated, truncated, info = super().step(2 * action.cpu().detach().numpy())
        ob[-1] /= 8.0  # 归一化角速度

        state = from_numpy(ob).float().reshape(-1)

        if self.history_buffer is not None:
            self.history_buffer.append(state)
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
        return torch.sum(torch.abs(state - tensor([1.0, 0.0, 0.0]))).item() < 1e-3

    def build_environment(self) -> Self:
        return self
