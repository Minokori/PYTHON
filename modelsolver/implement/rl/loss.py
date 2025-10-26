

from typing import TYPE_CHECKING, Literal

from torch import Tensor
from torch.nn.functional import mse_loss

from modelsolver.abc.rl.loss import IAgentLoss


class DefaultAgentLoss(IAgentLoss):

    if TYPE_CHECKING:
        def __call__(self, predicted: Tensor, label: Tensor | None = None, target: Literal["ddpg_actor", "ddpg_critic",
                                                                                           "behavior_clone"] | str = "ddpg_actor") -> Tensor:
            """由于强化学习智能体可能不同的部分需要使用不同的损失函数计算, 因此需要在调用时指定具体的目标"""
            ...

    def forward(self,
                predicted: Tensor,
                label: Tensor | None = None,
                target: Literal["ddpg_actor",
                                "ddpg_critic",
                                "behavior_clone"] | str = "ddpg_actor") -> Tensor:
        match target:
            case "ddpg_actor":
                return self.ddpg_actor_loss(predicted)
            case "ddpg_critic":
                if label is None:
                    raise ValueError("Critic loss requires label tensor")
                return self.ddpg_critic_loss(predicted, label)
            case "behavior_clone":
                assert label is not None, "Behavior cloning loss requires label tensor"
                return self.behavior_cloning_loss(predicted, label)
            case _:
                raise ValueError(f"Unknown target for loss computation: {target}")

    def behavior_cloning_loss(self, predicted: Tensor, label: Tensor) -> Tensor:
        return mse_loss(predicted, label)

    def ddpg_actor_loss(self, predicted: Tensor) -> Tensor:
        return -predicted.mean()

    def ddpg_critic_loss(self, predicted: Tensor, label: Tensor) -> Tensor:
        return mse_loss(predicted, label)
