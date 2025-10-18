from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import Module


class IModel(ABC, Module):
    """模型接口
    """

    if TYPE_CHECKING:
        def __call__(self, x: Sequence[Tensor] | Tensor, **kwargs) -> Tensor:
            """模型前向传播

            Args:
                x (Sequence[Tensor]): 批量样本数据

            Returns:
                output (Any): 批量的模型输出
            """
            ...

    @property
    @abstractmethod
    def name_for_save(self) -> str:
        """模型保存的名称"""
        ...
