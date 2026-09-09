"""数据模块, 定义了数据集接口、数据处理器接口、数据加载器接口以及强化学习相关的经验回放池接口"""
# region imports
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Self

from torch import Tensor
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
    def __add__(self, other: Self) -> Self:
        raise NotImplementedError("没有为 IDataset 实现 __add__ 方法")


class IDataProcesser(ABC):
    """数据处理器接口

    需要重写的方法:
    + `collate_fn` : 传递给 DataLoader 的 `collate_fn` 方法
    + `preprocess` : 预期接受 Dataloader 的输出, 并进行预处理
    + `postprocess` : 预期接受 IModel 的输出, 进行后处理, 以便可视化或其他用途
    """
    @abstractmethod
    def collate_fn(self, batch: Sequence) -> tuple[list[Tensor], list[Tensor]]:
        """传递给 DataLoader 的 `collate_fn` 方法.

        期望返回 输入, 目标, 以便于后续的训练和评估.
        """

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
    """对 `torch.utils.data.DataLoader` 的封装. (*仅添加了类型注解防止报错*)"""
    if TYPE_CHECKING:
        def __iter__(self) -> Iterator[tuple[Tensor, ...]]: ...
        def __next__(self) -> tuple[Tensor, ...]: ...


# region 强化学习相关接口
class IReplayBuffer(IDataset, ABC):
    """经验回放池接口

    需要重写的方法:
    + `__init__` : 构造函数, 接受 ReplayBufferConfig 作为参数
    + `__add__` (可选) : 合并两个经验回放池
    + `__len__` : 返回当前池中存储的**序列**数
    + `__getitem__` : 根据索引返回状态转移链条 `(s,a,r,s',done)`
    + `append` : 向池中添加一条序列 `(s,a,r,s')`
    + `sample` : 从经验回放池中随机采样一批数据
    需要重写的属性:
    + `can_sample` : 是否可以从池中采样
    + `config` : 经验回放池的配置

    """
    @property
    def config(self) -> ReplayBufferConfig:...

    def __init__(self, config: ReplayBufferConfig):
        ...

    def __add__(self, other: Self) -> Self:
        raise NotImplementedError("没有为 IReplayBuffer 实现 __add__ 方法")

    def __len__(self) -> int:
        """返回当前池中存储的序列数"""
        ...

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """根据索引返回状态转移链条

        Args:


        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 状态转移链条 (s,a,r,s',done)
        """
        ...
    # region 属性
    @property
    def can_sample(self) -> bool:
        """是否可以从池中采样"""
        ...
    # endregion

    @abstractmethod
    def append(
            self,
            state: Tensor,
            action: Tensor,
            reward: Tensor,
            next_state: Tensor,
            done: Tensor,
            new: bool = False) -> None:
        """向池中添加一条序列 `(s,a,r,s')` , 每个元素 shape = (1, dim)

        Args:
            state (Tensor): 状态 s
            action (Tensor): 在状态 s 下, 策略网络(Actor)输出的动作 a
            reward (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的奖励 r
            next_state (Tensor): 在状态 s 下, 执行动作 a 后, 环境返回的下一个状态 s'
            done (Tensor): 是否终止
            new (bool): 是否为新轨迹的开始. 默认为 False, 即默认添加到当前轨迹中. 设置为 True 时, 将在池中添加一条新轨迹.
        """
        ...




    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """从经验回放池中随机采样一批数据

        Returns:
            批量马尔可夫链 (tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): 状态、动作、奖励、下一个状态、终止标志 , shape = (batch, channel)
        """
        ...



class IExpertStore(ABC):
    """专家数据存储接口

    仅负责:
    1) 接收 DataFrame(state/action)
    2) 构建tensor缓存
    3) 按当前state_part做top-k最近邻goal采样

    需要重写:
    + `__init__` : 构造函数, 接受 ReplayBufferConfig 作为参数
    + `sample_expert_states` : 对batch中每个当前state, 找到expert中最近的top-k, 在top-k中随机选1个goal
    """
    @abstractmethod
    def __init__(self, config:ReplayBufferConfig) -> None:...

    @property
    def config(self) -> ReplayBufferConfig:...


    @abstractmethod
    def sample_expert_states(self, batch_state: Tensor, topk: int) -> tuple[Tensor, Tensor]:
        """对batch中每个当前state:
        - 找到expert中最近的top-k
        - 在top-k中随机选1个goal

        Args:
            batch_state (Tensor): 传入的状态，shape = (B, S)
            topk (int): 每个输入状态对应的最近邻专家状态数量.

        Returns:
            expert_ob (Tensor): 采样到的专家观测, shape = (B, S/2)
            chosen_indices (Tensor): 被选中的专家状态的索引, shape = (B,)
        """
# endregion
