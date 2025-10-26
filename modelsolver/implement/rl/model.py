
from torch import Tensor

from modelsolver.abc.rl.model import IActor, ICritic


class NullActor(IActor):
    """占位Actor网络, 基于 Q 学习的网络可能不需要策略网络"""

    def __init__(self):
        super().__init__()

    def forward(self, state: Tensor) -> Tensor:
        raise NotImplementedError("占位Actor网络不实现forward方法")


class NullCritic(ICritic):
    """占位Critic网络, 基于策略的网络可能不需要值网络"""

    def __init__(self):
        super().__init__()

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        raise NotImplementedError("占位Critic网络不实现forward方法")
