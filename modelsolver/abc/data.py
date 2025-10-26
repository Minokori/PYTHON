from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Iterator, Self

from torch import Tensor
from torch.utils.data import DataLoader

from modelsolver.abc.config import DataConfig


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

# TODO 抽成 dict 形式 __getitem__(self, mode="调用的函数")-> Callable
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
