from abc import ABC, abstractmethod
from typing import Literal, Self

import torch
from numpy import bool_, float32, ndarray
from numpy.typing import NDArray
from torch import Tensor

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
            state: Tensor | NDArray[float32],
            action: Tensor | NDArray[float32],
            reward: Tensor | NDArray[float32] | float,
            next_state: Tensor | NDArray[float32],
            done: Tensor | NDArray[bool_] | bool) -> None:
        """向池中添加一条序列 `(s,a,r,s')` , 每个元素 shape = (1, dim)

        Args:
            state (Tensor): 状态 s
            action (Tensor): 在状态 s 下, 策略网络(Actor)输出的动作 a
            reward (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的奖励 r
            next_state (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的下一个状态 s'
            done (Tensor): 是否终止
        """
        match state:
            case Tensor():
                state = state.detach().cpu()
            case ndarray():
                state = torch.from_numpy(state).float()
        self.state_buffer[self.state_index] = state.cpu()
        self.state_index = (self.state_index + 1) % self.config.capacity

        match action:
            case Tensor():
                action = action.detach().cpu()
            case ndarray():
                action = torch.from_numpy(action).float()
        self.action_buffer[self.action_index] = action.cpu()
        self.action_index = (self.action_index + 1) % self.config.capacity

        match reward:
            case Tensor():
                reward = reward.detach().cpu()
            case ndarray():
                reward = torch.from_numpy(reward).float()
            case float():
                reward = torch.tensor(reward, dtype=torch.float32).reshape(-1, 1)
        self.reward_buffer[self.reward_index] = reward.cpu()  # type: ignore
        self.reward_index = (self.reward_index + 1) % self.config.capacity

        match next_state:
            case Tensor():
                next_state = next_state.detach().cpu()
            case ndarray():
                next_state = torch.from_numpy(next_state).float()
        self.next_state_buffer[self.next_state_index] = next_state.cpu()
        self.next_state_index = (self.next_state_index + 1) % self.config.capacity

        match done:
            case Tensor():
                done = done.detach().cpu()
            case ndarray():
                done = torch.from_numpy(done).float()
            case bool():
                done = torch.tensor(done, dtype=torch.bool).reshape(-1, 1)
        self.done_buffer[self.done_index] = done.cpu()
        self.done_index = (self.done_index + 1) % self.config.capacity

    def _create_buffer(self):
        self.state_buffer = torch.zeros((self.config.capacity, self.config.state_dim), dtype=torch.float32)
        self.action_buffer = torch.zeros((self.config.capacity, self.config.action_dim), dtype=torch.float32)
        self.reward_buffer = torch.zeros((self.config.capacity, 1), dtype=torch.float32)
        self.next_state_buffer = torch.zeros((self.config.capacity, self.config.state_dim), dtype=torch.float32)
        self.done_buffer = torch.zeros((self.config.capacity, 1), dtype=torch.bool)

        self.state_index = 0
        self.action_index = 0
        self.reward_index = 0
        self.next_state_index = 0
        self.done_index = 0

    def get_action_param(self) -> tuple[Tensor, Tensor]:
        mean = torch.mean(self.action_buffer, dim=0).detach()
        std = torch.std(self.action_buffer, dim=0).detach()
        return mean, std


class IEnvironmentConverter(ABC):
    """在 仿真环境的 (s,a) | 模型可接受的 (s,a) | 真实环境的 (s,a) 之间转换数据 的接口
    """

    @abstractmethod
    def convert(self,
                datatype: Literal["s", "a"],
                data: NDArray[float32] | Tensor,
                from_: Literal["simulation", "realworld", "model"],
                to_: Literal["simulation", "realworld", "model"]) -> Tensor:
        """转换数据

        将 {datatype} 从 适用于{from_}的数据格式 转换为 适用于{to_}的数据格式, 数据具体内容为 {data}

        *e.g.: 将 状态s 从 适用于真实世界的格式 (x,y,z,v,heading) 转换为 适用于仿真模型的格式 (x,y,v,heading), 具体内容为 (3.00,4.00,0.00, 90)*
        Args:
            datatype (Literal[&quot;s&quot;, &quot;a&quot;]): _description_
            data (NDArray[float32] | Tensor): _description_
            from_ (Literal[&quot;simulation&quot;, &quot;realworld&quot;, &quot;model&quot;]): _description_
            to_ (Literal[&quot;simulation&quot;, &quot;realworld&quot;, &quot;model&quot;]): _description_

        Returns:
            _description_ (Tensor): _description_
        """
