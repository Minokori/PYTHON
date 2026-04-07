"""强化学习奖励函数接口"""

from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class IReward(ABC, Module):

    def __init__(self):
        super().__init__()

    def forward(self, state: Tensor, action: Tensor, next_state: Tensor, **kwargs) -> Tensor:
        """计算奖励

        Args:
            state (Tensor): 当前状态, shape = (B, state_dim)
            action (Tensor): 当前动作, shape = (B, action_dim)
            next_state (Tensor): 下一状态, shape = (B, state_dim)
            kwargs: 额外的参数, shape = (B, ...)

        Returns:
            reward (Tensor): 奖励, shape = (B,)
        """
        ...

    @property
    @abstractmethod
    def is_learnable(self) -> bool:
        """是否可学习 (IRL 中可能需要学习奖励函数)"""
        ...

