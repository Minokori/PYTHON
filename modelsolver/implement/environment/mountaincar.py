# regiopn imports
import logging
from typing import Self

import numpy as np
import torch
from gymnasium.envs.classic_control import Continuous_MountainCarEnv
from torch import Tensor, concat, from_numpy, no_grad, tanh, tensor

from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.reward import IReward


logging.basicConfig(level=logging.DEBUG, filename="mountaincar.log", filemode="w",encoding="utf-8")
np.set_printoptions(precision=2, suppress=True)
# endregion




class MountainCarEnvironment(Continuous_MountainCarEnv,IEnvironment):
    """山地车环境"""
    @property
    def ZERO_ACTION(self) -> Tensor:
        return tensor([0.0])
    @property
    def GOAL(self) -> Tensor:
        return tensor([(0.45+0.3)/0.9, 0.0, 0.57])  # 位置(归一到[-1,1]), 速度, 高度

    def __init__(self) -> None:
        super().__init__(render_mode="human")
        self.__timestep = 0
        open("mountaincar.log", "w").close()  # 清空日志文件

    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, info = super().reset()
        height = tensor(self._height(ob[0])).float().reshape(1)
        self.__timestep = 0

        # state
        state = self.observation_to_state(ob)

        reward, done = self.compute_reward(state, self.ZERO_ACTION, state)
        return state, reward, done, False, info


    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, _, _, truncated, info = super().step(action.cpu().detach().numpy())
        height = tensor(self._height(ob[0])).float().reshape(1)
        self.__timestep += 1

        # state
        state = self.observation_to_state(ob)

        reward, done = self.compute_reward(state, action, state)
        return state, reward, done, truncated, info

    @no_grad
    def compute_reward(self, state: Tensor, action: Tensor, next_state: Tensor) -> tuple[Tensor, Tensor]:
        # state: (position, velocity, height, goal_position, goal_velocity, goal_height)
        # 奖励函数设计: 在谷底, 高度, 速度, 都变化缓慢, 难以学习.

        # 高度奖励: 当智能体在谷底时, 给予较小的奖励, 当智能体逐渐爬升时, 给予更大的奖励, 鼓励智能体克服山谷.
        height_reward = state[2]*2

        # 位置奖励: 当智能体接近目标位置时给予正奖励, 当智能体远离目标位置时给予负奖励, 鼓励智能体向目标位置移动.
        position_reward = state[0]

        # 成功奖励: 当智能体成功到达目标位置时给予大量奖励, 鼓励智能体完成任务.
        terminated = tensor((state[0] >= self.GOAL[0]) and (state[1] >= self.GOAL[1])).float()  # 原环境的判断逻辑
        success_reward = terminated * 500

        # 速度奖励: 当速度较大时给予正奖励, 当速度较小时给予负奖励, 鼓励智能体保持较高的速度以克服山谷.
        speed_reward = abs(state[1]) if abs(state[1]) > 0.2 else -abs(state[1])*5
        # 加速度奖励: 当加速度方向与当前速度方向一致时给予正奖励, 否则给予负奖励, 鼓励智能体采取有助于增加速度的行动.
        acc_reward = -1 if action.cpu().detach() * state[1] <= 0 else 1


        # 设计权重: 高度较低时以速度/加速度奖励为主, 高度较高时以位置/成功奖励为主.
        # 这样可以鼓励智能体在谷底时保持较高的速度, 在爬升过程中逐渐转向位置奖励, 最终在接近目标位置时获得成功奖励.
        dynamic_weight = 1-state[2].item()
        goal_weight = state[2].item()



        reward = tanh(
                (height_reward + position_reward) * goal_weight+\
                 (acc_reward + speed_reward)*dynamic_weight)
        reward +=success_reward
        return reward, terminated

    @no_grad
    def observation_to_state(self, observation: np.ndarray) -> Tensor:
        # observation: (position, velocity)
        # state: (position, velocity, height, goal_position, goal_velocity, goal_height)
        height = tensor(self._height(observation[0])).float().reshape(1)
        state = from_numpy(observation.copy()).float().reshape(-1)
        state[0] = (state[0] + 0.3) / 0.9  # 位置归一化到 [-1, 1]
        state[1] = (state[1] / 0.07) # 速度归一化到 [-1, 1]
        state = concat([state, height, self.GOAL])  # state 中包含当前高度和目标高度, 以便奖励函数计算
        return state

    def build_environment(self) -> Self:
        return self


class MountainCarReward(IReward):

    def forward(self,**kwargs:Tensor) -> tuple[Tensor, Tensor]:
        state = kwargs["state"]  # (B,6)
        action = kwargs["action"] # (B,1)
        next_state = kwargs["next_state"] # (B,6)
        goal = state[:,3:]  # (B,3)
        # state: (position, velocity, height, goal_position, goal_velocity, goal_height)
        # 奖励函数设计: 在谷底, 高度, 速度, 都变化缓慢, 难以学习.

        # 高度奖励: 当智能体在谷底时, 给予较小的奖励, 当智能体逐渐爬升时, 给予更大的奖励, 鼓励智能体克服山谷.
        height_reward = state[:,2:3]*2 # (B,1)

        # 位置奖励: 当智能体接近目标位置时给予正奖励, 当智能体远离目标位置时给予负奖励, 鼓励智能体向目标位置移动.
        position_reward = state[:,0:1] # (B,1)

        # 成功奖励: 当智能体成功到达目标位置时给予大量奖励, 鼓励智能体完成任务.
        terminated = tensor((state[:, 0:1] >= goal[:, 0:1])).float()  # 原环境的判断逻辑


        # 速度奖励: 当速度较大时给予正奖励, 当速度较小时给予负奖励, 鼓励智能体保持较高的速度以克服山谷.
        # speed_reward = abs(state[:,1]) if abs(state[:,1]) > 0.2 else -abs(state[:,1])*5  # (B,1)
        speed_reward = torch.where(abs(state[:,1]) > 0.2, abs(state[:,1]), -abs(state[:,1])*5).reshape(-1,1)

        # 加速度奖励: 当加速度方向与当前速度方向一致时给予正奖励, 否则给予负奖励, 鼓励智能体采取有助于增加速度的行动.
        acc_reward = torch.where(
            (action.detach().cpu().view(-1) * state[:, 1].detach().cpu().view(-1)) <= 0,
            -torch.ones(action.numel(), dtype=action.dtype),
            torch.ones(action.numel(), dtype=action.dtype)
        ).float().reshape(-1,1)

        # 设计权重: 高度较低时以速度/加速度奖励为主, 高度较高时以位置/成功奖励为主.
        # 这样可以鼓励智能体在谷底时保持较高的速度, 在爬升过程中逐渐转向位置奖励, 最终在接近目标位置时获得成功奖励.
        dynamic_weight = 1-state[:,2:3]
        goal_weight = state[:,2:3]



        reward = tanh(
                (height_reward + position_reward) * goal_weight+\
                 (acc_reward + speed_reward)*dynamic_weight)
        return reward, terminated

    @property
    def is_learnable(self) -> bool:
        return False