"""数据模块, 定义了数据集接口、数据处理器接口、数据加载器接口以及强化学习相关的经验回放池接口"""
# region imports
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Self

from torch import Tensor, randint, stack
from torch.utils.data import DataLoader

from modelsolver.abc.config import DataConfig, ReplayBufferConfig


# endregion

class IDataset(ABC):
    """对数据集的抽象接口

    需要重写的方法:
    + `__init__` : 构造函数, 接受 DataConfig 作为参数
    + `__getitem__` : 根据索引返回数据
    + `__len__` : 返回数据集的长度
    + `__add__` (可选) : 合并两个数据集
    """
    @abstractmethod
    def __init__(self, config: DataConfig): ...
    @abstractmethod
    def __getitem__(self, index: int | list[int] | slice) -> Sequence: ...
    @abstractmethod
    def __len__(self) -> int: ...
    def __add__(self, other: Self) -> Self: raise NotImplementedError("没有为 IDataset 实现 __add__ 方法")


class IDataProcesser(ABC):
    """数据处理器接口"""
    @abstractmethod
    def collate_fn(self, batch: Sequence) -> tuple[list[Tensor], list[Tensor]]:
        """传递给 DataLoader 的 collate_fn 函数"""

    @abstractmethod
    def preprocess(self, batch: Sequence, **kwargs) -> tuple[Tensor, ...]:
        """预期接受 Dataloader 的输出, 并进行预处理

        >>> for batch in dataloader:
                inputs, targets = self.preprocess(batch)
                outputs = model(inputs)

        预期得到 Tensor
        """

    @abstractmethod
    def postprocess(self, batch: Sequence[Tensor] | Tensor, **kwargs) -> Any:
        """预期接受 IModel 的输出, 进行后处理, 以便可视化或其他用途"""


class IDataLoader(DataLoader):
    """对 `torch.utils.data.DataLoader` 的封装. *添加了类型注解防止报错*"""
    if TYPE_CHECKING:
        def __iter__(self) -> Iterator[tuple[Tensor, ...]]: ...
        def __next__(self) -> tuple[Tensor, ...]: ...


# region 强化学习相关接口
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
        return len(self._state_buffer)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """根据索引返回状态转移链条

        Args:


        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 状态转移链条 (s,a,r,s',done)
        """
        return (self._state_buffer[index],
                self._action_buffer[index],
                self._reward_buffer[index],
                self._next_state_buffer[index],
                self._done_buffer[index])
    # region 属性
    @property
    def can_sample(self) -> bool:
        """是否可以从池中采样"""
        return len(self._state_buffer) >= self.config.minimal_capacity
    # endregion

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
        self._state_buffer.append(state.cpu().detach().float().reshape(-1))
        self._action_buffer.append(action.cpu().detach().float().reshape(-1))
        self._reward_buffer.append(reward.cpu().detach().float().reshape(-1))
        self._next_state_buffer.append(next_state.cpu().detach().float().reshape(-1))
        self._done_buffer.append(done.cpu().detach().float().reshape(-1))




    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """从经验回放池中随机采样一批数据

        Returns:
            批量马尔可夫链 (tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): 状态、动作、奖励、下一个状态、终止标志 , shape = (batch, channel)
        """
        if len(self._state_buffer) < self.config.batch_size:
            raise ValueError("Not enough samples in replay buffer")
        indices = randint(0, len(self._state_buffer), (self.config.batch_size,))
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

    def _create_buffer(self):
        """创建经验回放池的缓冲区"""
        self._state_buffer: deque[Tensor] = deque(maxlen=self.config.capacity)
        """状态 s 的缓冲区"""
        self._action_buffer: deque[Tensor] = deque(maxlen=self.config.capacity)
        """动作 a 的缓冲区"""
        self._reward_buffer: deque[Tensor] = deque(maxlen=self.config.capacity)
        """奖励 r 的缓冲区"""
        self._next_state_buffer: deque[Tensor] = deque(maxlen=self.config.capacity)
        """下一个状态 s' 的缓冲区"""
        self._done_buffer: deque[Tensor] = deque(maxlen=self.config.capacity)
        """终止标志 done 的缓冲区"""

# endregion
