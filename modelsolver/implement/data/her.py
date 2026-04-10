"""Hindsight Experience Replay (HER) 实现"""


from dataclasses import dataclass
from typing import NamedTuple

from dataclasses_json import dataclass_json

import torch
from torch import Tensor

from modelsolver.abc.config import ReplayBufferConfig
from modelsolver.abc.data import IReplayBuffer



# TODO 将HER配置移出ReplayBufferConfig


@dataclass_json
@dataclass
class HEReplayConfig(ReplayBufferConfig):
    """HER经验回放池配置"""
    her_p: float = 0.8  # HER采样的概率
    """HER采样的概率, 即在采样时, 以 her_p 的概率使用HER算法生成新的样本, 否则使用原始样本"""
    her_threshold: float = 0.5  # HER奖励的距离阈值
    """HER奖励的距离阈值, 即在使用HER算法生成新的样本时, 如果T.step的s'和goal的距离小于该阈值, 则给予正奖励, 否则给予负奖励"""
    her_reward:tuple[float, float] = (-1.0, 1.0)  # HER奖励设计, (负奖励, 正奖励)
    """HER奖励设计, (负奖励, 正奖励)"""


# TODO 将class 定义移出her.py

class Trajectory(NamedTuple):
    """一条轨迹, 包含状态链、动作链、奖励链、下一状态链和终止标志链"""
    states: list[Tensor]  # shape (T, state_dim *2) 包含原状态和goal
    """状态链, shape = [T*, state_dim * 2]

    **包含原状态s和目标g, 以便HER算法的实现**
    """
    actions: list[Tensor]  # shape (T, action_dim)
    """动作链, shape = [T*, action_dim]"""
    rewards: list[Tensor]  # shape (T,)
    """奖励链, shape = [T*, 1]"""
    next_states: list[Tensor]  # shape (T, state_dim *2) 包含原状态和goal
    """下一状态链, shape = [T*, state_dim * 2]

    **包含原状态s和目标g, 以便HER算法的实现**
    """
    dones: list[Tensor]  # shape (T,)
    """终止标志链, shape = [T*, 1]"""

    def __getitem__(self, index:int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        s = self.states[index]
        a = self.actions[index]
        r = self.rewards[index]
        s_ = self.next_states[index]
        d = self.dones[index]

        return s, a, r, s_, d

    def __len__(self) -> int:
        return len(self.states)

    def append(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor) -> None:
        """向轨迹中添加一个时间步的数据

        Args:
            state (Tensor): 状态 s
            action (Tensor): 在状态 s 下, 策略网络(Actor)输出的动作 a
            reward (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的奖励 r
            next_state (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的下一个状态 s'
            done (Tensor): 是否终止
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def pop(self, index: int) -> None:
        """从轨迹中删除一个时间步的数据

        Args:
            index (int): 要删除的时间步的索引
        """
        self.states.pop(index)
        self.actions.pop(index)
        self.rewards.pop(index)
        self.next_states.pop(index)
        self.dones.pop(index)


class SimpleHEReplay(IReplayBuffer):
    """简单的HER实现 (没有使用环境的reward函数，而是直接计算HER奖励)"""

    def __init__(self, config:ReplayBufferConfig):
        assert issubclass(type(config), HEReplayConfig), "HER 经验回放池需要 HEReplayConfig 实例作为配置"
        self._config = config
        self._create_buffer()
        pass

    @property
    def config(self) -> HEReplayConfig:
        return self._config # type: ignore

    def __len__(self) -> int:
        return sum(len(traj) for traj in self.buffer)

    def _create_buffer(self):
        self.buffer: list[Trajectory] = []

    def append(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor, new:bool = False) -> None:
        """向池中添加一条序列 `(s,a,r,s')` , 每个元素 shape = (1, dim)

        Args:
            state (Tensor): 状态 s
            action (Tensor): 在状态 s 下, 策略网络(Actor)输出的动作 a
            reward (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的奖励 r
            next_state (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的下一个状态 s'
            done (Tensor): 是否终止
        """



        if new:
            # 添加新轨迹
            self.buffer.append(Trajectory(states=[state], actions=[action], rewards=[reward], next_states=[next_state], dones=[done]))
        else:
            self.buffer[-1].append(state, action, reward, next_state, done)

        # 达到容量时, 删除最旧的轨迹
        if len(self) == self._config.capacity:

            # 若仅有一条轨迹, 无法丢弃, 删除轨迹中最早的一个时间步
            if len(self.buffer) ==1:
                self.buffer[0].pop(0)
                self.buffer[-1].append(state, action, reward, next_state, done)
            else:
                self.buffer.pop(0)  # 删除最旧的轨迹
    @property
    def can_sample(self) -> bool:
        """是否可以从池中采样"""
        return len(self) > self._config.minimal_capacity

    def sample(self, her:bool=True) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        states = []
        actions = []
        rewards = []
        next_states=[]
        dones = []

        for _ in range(self.config.batch_size):
            trajectory = self.buffer[int(torch.randint(0,len(self.buffer),(1,)))]  # 随机采样一条轨迹

            # 随机采样一个时间步, 记为 T.now (变量state)
            idx_now = int(torch.randint(0, len(trajectory), (1,)))  # 采样的索引

            state, action, reward, next_state, done = trajectory[idx_now]


            if her and torch.rand(1).item()<self.config.her_p:
                idx_goal = int(torch.randint(idx_now, len(trajectory), (1,)))  # 采样的索引
                goal_state, goal_reward, goal_action, goal_next_state, goal_done = trajectory[idx_goal]

                g = self.config.state_dim //2

                diff = torch.sum(torch.norm(
                    goal_state[g:] - next_state[g:],p=0.5
                )).item()


                # TODO 用 IReward 进行注入.
                generated_reward = self.config.her_reward[0] if diff > self.config.her_threshold else self.config.her_reward[1]  # 为什么是大于?
                generated_done = False if diff > self.config.her_threshold else True  #为什么是大于?

                generated_state = torch.cat((state[:g], goal_state[g:]), dim=0)
                generated_next_state = torch.cat((next_state[:g], goal_state[g:]), dim=0)

                states.append(generated_state)
                actions.append(action)
                rewards.append(torch.tensor([generated_reward]))
                next_states.append(generated_next_state)
                dones.append(torch.tensor([generated_done]).float())

            else:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done.reshape(1))

                pass
        return torch.stack(states), torch.stack(actions), torch.stack(rewards), torch.stack(next_states), torch.stack(dones)

