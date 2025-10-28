from abc import ABC
from collections import deque
from typing import Self

import torch
from torch import Tensor, no_grad

from modelsolver.abc.data import IDataset
from modelsolver.abc.rl.config import ReplayBufferConfig


class IReplayBuffer(IDataset, ABC):
    """经验回放池接口

    *IReplayBuffer 已经有简单的实现, 可以不用重载*
    """

    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self._create_buffer()

    def __add__(self, other: Self) -> Self:
        raise NotImplementedError("没有为 IReplayBuffer 实现 __add__ 方法")

    def __len__(self) -> int:
        """返回当前池中存储的序列数"""
        return len(self.state_buffer)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """根据索引返回状态转移链条

        Args:


        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 状态转移链条 (s,a,r,s',done)
        """
        return (self.state_buffer[index],
                self.action_buffer[index],
                self.reward_buffer[index],
                self.next_state_buffer[index],
                self.done_buffer[index])
    # region 属性

    def append(
            self,
            state: Tensor,
            action: Tensor,
            reward: Tensor,
            next_state: Tensor,
            done: Tensor) -> None:
        """向池中添加一条序列 `(s,a,r,s')` , 每个元素 shape = (1, dim)

        Args:
            state (Tensor): 状态 s
            action (Tensor): 在状态 s 下, 策略网络(Actor)输出的动作 a
            reward (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的奖励 r
            next_state (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的下一个状态 s'
            done (Tensor): 是否终止
        """
        self.state_buffer.append(state.cpu().detach().float().reshape(1, -1))
        self.action_buffer.append(action.cpu().detach().float().reshape(1, -1))
        self.reward_buffer.append(reward.cpu().detach().float().reshape(1, -1))
        self.next_state_buffer.append(next_state.cpu().detach().float().reshape(1, -1))
        self.done_buffer.append(done.cpu().detach().float().reshape(1, -1))



    def _create_buffer(self):
        self.state_buffer = deque(maxlen=self.config.capacity)
        self.action_buffer = deque(maxlen=self.config.capacity)
        self.reward_buffer = deque(maxlen=self.config.capacity)
        self.next_state_buffer = deque(maxlen=self.config.capacity)
        self.done_buffer = deque(maxlen=self.config.capacity)

    @no_grad()
    def get_action_mean_std(self) -> tuple[Tensor, Tensor]:
        mean = torch.mean(torch.stack(list(self.action_buffer)), dim=0).detach()
        std = torch.std(torch.stack(list(self.action_buffer)), dim=0).detach()
        return mean, std

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if len(self.state_buffer) < self.config.batch_size:
            raise ValueError("Not enough samples in replay buffer")
        indices = torch.randint(0, len(self.state_buffer), (self.config.batch_size,))
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for idx in indices:
            state, action, reward, next_state, done = self[int(idx.item())]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.stack(dones)
