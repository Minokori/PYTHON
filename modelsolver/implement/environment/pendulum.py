# regiopn imports
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium.envs.classic_control import PendulumEnv
from torch import Tensor, concat, from_numpy, no_grad, tensor

from modelsolver.abc.config import EnvironmentConfig
from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.reward import IReward


logging.basicConfig(level=logging.DEBUG, filename="plog.log", filemode="w",encoding="utf-8")
np.set_printoptions(precision=2, suppress=True)
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

    def __init__(self, config: EnvironmentConfig, reward: IReward ) -> None:
        PendulumEnv.__init__(self, render_mode="human")
        self._reward_function = reward
        self.history_buffer = deque(maxlen=config.terminated_delta) if config.terminated_delta > 0 else None
        """历史状态缓冲区. 不启用终止状态时, 为 None"""
        self.truncated_time = config.truncated_time if config.truncated_time > 0 else None
        """时间步截断时间. 不启用截断时, 为 None"""
        self.time = 0 if config.truncated_time > 0 else None
        """当前仿真时间. 不启用截断时, 为 None"""

    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        self.time = 0 if self.time is not None else None
        ob, info = super().reset()
        state= from_numpy(ob.copy()).float().reshape(-1)
        state[-1]/= 8.0  # 归一化角速度
        state = concat([state, self.GOAL])
        reward, done = self._reward_function(state=state, action=self.ZERO_ACTION, next_state=state)
        return state, reward, tensor(0).float().reshape(1), False, info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, reward, terminated, truncated, info = super().step(2 * action.cpu().detach().numpy())
        state = from_numpy(ob.copy()).float().reshape(-1)
        state[-1]/= 8.0  # 归一化角速度
        state = concat([state, self.GOAL])
        if self.history_buffer is not None:
            self.history_buffer.append(state)
        if self.time is not None:
            self.time += 1

        angle = np.rad2deg(np.arctan2(ob[1], ob[0]))
        text = "↻" if action[0]>0 else "↺"
        logging.debug(f"动作:{text}{abs(action[0]):.2f}, 角度:{angle:.2f}, 角速度:{ob[-1]:.2f}  奖励:{reward:.2f}")


        reward,done = self._reward_function(state=state, action=action, next_state=state)
        return state, reward, self.is_terminated(), self.is_truncated(), info


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
        return torch.sum(torch.abs(state[0:3] - tensor([1.0, 0.0, 0.0]))).item() < 1e-3


    @property
    def ZERO_ACTION(self) -> Tensor:
        return tensor([0.0]).float().cpu()

    @property
    def GOAL(self) -> Tensor:
        return tensor([1.0, 0.0, 0.0]).float().cpu()

class PendulumReward(IReward):

    WEIGHT = tensor([1.0, 0.8, 0.2]).float().cpu().reshape(-1,1)

    @no_grad
    def forward(self, **kwargs:Tensor) -> tuple[Tensor, Tensor]:
        # state&goal: (cos, sin, v_theta)
        state = kwargs["state"]  # shape = (B, state_dim)
        batched = len(state.shape) > 1
        state = state.reshape(-1,6)
        action = kwargs["action"].reshape(-1,1)  # shape = (B, action_dim)
        next_state = kwargs["next_state"].reshape(-1,6)  # shape = (B, state_dim)




        reward = -((next_state[:,:3] - next_state[:,3:]).abs() @ self.WEIGHT).reshape(-1,1)
        done = (reward > -0.1).float().reshape(-1, 1)

        reward += torch.where((state[:,0:1]+1).abs()< 0.2, -2, 0)  # 当cos接近-1时, 给予较大的负奖励, 以鼓励智能体远离下垂位置

        reward += done * 3  # 当达到目标位置时, 给予额外奖励

        if not batched:
            reward = reward.reshape(1)
            done = done.reshape(1)

        return reward, done

    @property
    def is_learnable(self) -> bool:
        return False