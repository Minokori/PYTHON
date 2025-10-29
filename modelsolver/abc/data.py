from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Self

from torch import Tensor, mean, no_grad, randint, stack, std
from torch.utils.data import DataLoader

from modelsolver.abc.config import DataConfig, ReplayBufferConfig


class IDataset(ABC):
    """对数据集的抽象接口
    """
    @abstractmethod
    def __init__(self, config: DataConfig): ...
    @abstractmethod
    def __getitem__(self, index: int | list[int] | slice) -> Sequence: ...
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __add__(self, other: Self) -> Self: ...

class IDataProcesser(ABC):
    @abstractmethod
    def collate_fn(self, batch: Sequence) -> tuple[list[Tensor], list[Tensor]]:
        """传递给 DataLoader 的 collate_fn 函数"""

    @abstractmethod
    def preprocess(self, batch: Sequence, **kwargs) -> tuple[Tensor, ...]:
        """预期接受 Dataloader 的输出, 并进行预处理

        >>> for batch in dataloader:
                inputs, targets = self.preprocess(batch)
                outputs = model(inputs)

        预期得到 Tensor (在 cuda 上)
        """

    @abstractmethod
    def postprocess(self, batch: Sequence[Tensor] | Tensor, **kwargs) -> Any:
        """预期接受 IModel 的输出, 进行后处理, 以便可视化或其他用途"""


class IDataLoader(DataLoader):
    """对 `torch.utils.data.DataLoader` 的封装. 添加了类型注解防止报错"""
    if TYPE_CHECKING:
        def __iter__(self) -> Iterator[tuple[Tensor, ...]]: ...
        def __next__(self) -> tuple[Tensor, ...]: ...


# region RL
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

    @property
    def can_sample(self) -> bool:
        """是否可以从池中采样"""
        return len(self.state_buffer) >= self.config.minimal_capacity

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
        self.state_buffer.append(state.cpu().detach().float().reshape(-1))
        self.action_buffer.append(action.cpu().detach().float().reshape(-1))
        self.reward_buffer.append(reward.cpu().detach().float().reshape(-1))
        self.next_state_buffer.append(next_state.cpu().detach().float().reshape(-1))
        self.done_buffer.append(done.cpu().detach().float().reshape(-1))

    def _create_buffer(self):
        self.state_buffer = deque(maxlen=self.config.capacity)
        self.action_buffer = deque(maxlen=self.config.capacity)
        self.reward_buffer = deque(maxlen=self.config.capacity)
        self.next_state_buffer = deque(maxlen=self.config.capacity)
        self.done_buffer = deque(maxlen=self.config.capacity)

    @no_grad()
    def get_action_mean_std(self) -> tuple[Tensor, Tensor]:
        action_mean = mean(stack(list(self.action_buffer)), dim=0).detach()
        action_std = std(stack(list(self.action_buffer)), dim=0).detach()
        return action_mean, action_std

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if len(self.state_buffer) < self.config.batch_size:
            raise ValueError("Not enough samples in replay buffer")
        indices = randint(0, len(self.state_buffer), (self.config.batch_size,))
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
        return stack(states), stack(actions), stack(rewards), stack(next_states), stack(dones)

# endregion
