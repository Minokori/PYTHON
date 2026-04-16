"""强化学习奖励函数接口"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import Module


class IReward(ABC, Module):

    if TYPE_CHECKING:
        def __call__(self, **kwargs:Tensor) -> tuple[Tensor, Tensor]:
            """计算奖励

            奖励函数可能传入两种张量大小:

            + 和环境交互时, 传入的张量形状为: (特征维度, ). *例如, 传入的奖励形状为 (1,)*
            + 从 HER 经验回放中采样时, 传入的张量形状为: (批量大小, 特征维度). *例如, 传入的奖励形状为 (B, 1)*

            在返回 (奖励, terminated) 时, 需要保证奖励和 terminated 的形状与输入的张量形状一致.
            *例如, 当输入奖励形状为 (B, 1) 时, 返回的奖励和 terminated 也应该是 (B, 1)*
            """
            ...

    def __init__(self):
        super().__init__()

    def forward(self,  **kwargs:Tensor) -> tuple[Tensor, Tensor]:
        """计算奖励

        Args:
            state (Tensor): 当前状态, shape = (B, state_dim)
            action (Tensor): 当前动作, shape = (B, action_dim)
            next_state (Tensor): 下一状态, shape = (B, state_dim)
            kwargs: 额外的参数, shape = (B, ...)

        Returns:
            reward (Tensor): 奖励, shape = (B,)
            done (Tensor): 是否结束, shape = (B,)
        """
        ...

    @property
    @abstractmethod
    def is_learnable(self) -> bool:
        """是否可学习 (IRL 中可能需要学习奖励函数)"""
        ...

