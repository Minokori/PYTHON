"""预设的损失函数"""
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, mean
from torch.nn.functional import mse_loss
from torchmetrics import RelativeSquaredError

from modelsolver.abc.functional import IAgentLoss
from modelsolver.implement.loss.rse import RelativeSquaredErrorLoss


__all__ = ["RelativeSquaredErrorLoss"]

# region RL


class DefaultAgentLoss(IAgentLoss):
    """默认的智能体损失函数, 适用于大多数算法."""
    if TYPE_CHECKING:
        def __call__(self,
                     predicted: Tensor,
                     label: Tensor | None = None,
                     target: Literal["ddpg_actor", "ddpg_critic", "behavior_clone"] | str = "ddpg_actor") -> Tensor:
            """由于强化学习智能体可能不同的部分需要使用不同的损失函数计算, 因此需要在调用时指定具体的目标"""
            ...

    def __init__(self) -> None:
        super().__init__()
        self.rse_action: RelativeSquaredError = None  # type: ignore
        self.rse_state: RelativeSquaredError = None  # type: ignore

    def forward(self,
                predict: Tensor,
                label: Tensor | None = None,
                target: Literal["ddpg_actor",
                                "ddpg_critic",
                                "behavior_clone",
                                "irl"] | str = "ddpg_actor", **kwargs: Tensor) -> Tensor:
        match target:
            case "ddpg_actor":
                return self.ddpg_actor_loss(predict)
            case "ddpg_critic" | "sac_critic":
                assert label is not None, "Critic loss requires label tensor"
                return self.ddpg_critic_loss(predict, label)
            case "behavior_clone":
                assert label is not None, "Behavior cloning loss requires label tensor"
                return self.behavior_cloning_loss(predict, label)
            case "sac_actor":
                q = predict
                q_other = kwargs["q_other"]
                log_prob = kwargs["log_prob"]
                log_alpha = kwargs["log_alpha"]
                return self.sac_actor_loss(q, q_other, log_prob, log_alpha)
            case "irl":
                return self.irl_loss(predict)
            case _:
                raise ValueError(f"Unknown target for loss computation: {target}")

    def behavior_cloning_loss(self, predicted_action: Tensor, expert_action: Tensor) -> Tensor:
        B, C = predicted_action.shape
        if self.rse_action is None:
            self.rse_action = RelativeSquaredError(C).cuda()
        return self.rse_action(predicted_action, expert_action)

    def ddpg_actor_loss(self, predicted_q: Tensor) -> Tensor:
        return -mean(predicted_q)

    def ddpg_critic_loss(self, predicted_q: Tensor, target_q: Tensor) -> Tensor:
        # if self.rse_state is None:
        #     B,C = predicted_q.shape
        #     self.rse_state = RelativeSquaredError(C).cuda()
        # return self.rse_state(predicted_q, target_q)
        return mean(mse_loss(predicted_q, target_q))

    def irl_loss(self, predicted_q:Tensor)->Tensor:
        # TODO
        return -mean(predicted_q)

    # TODO kwargs
    def sac_actor_loss(self, predicted_q: Tensor, predicted_q_other: Tensor, log_prob: Tensor, log_alpha: Tensor) -> Tensor:
        alpha = log_alpha.detach().exp()
        return mean(alpha * log_prob - torch.min(predicted_q, predicted_q_other))

# endregion
