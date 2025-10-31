"""预设的损失函数"""
# isort: skip_file
from typing import TYPE_CHECKING, Literal
from modelsolver.abc.functional import IAgentLoss
from torch import Tensor, mean
import torch
from torchmetrics import RelativeSquaredError
from .rse import RelativeSquaredErrorLoss
from torch.nn.functional import mse_loss
__all__ = ["RelativeSquaredErrorLoss"]

# region RL


class DefaultAgentLoss(IAgentLoss):

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
                predicted: Tensor,
                label: Tensor | None = None,
                target: Literal["ddpg_actor",
                                "ddpg_critic",
                                "behavior_clone"] | str = "ddpg_actor", **kwargs: Tensor) -> Tensor:
        match target:
            case "ddpg_actor":
                return self.ddpg_actor_loss(predicted)
            case "ddpg_critic" | "sac_critic":
                assert label is not None, "Critic loss requires label tensor"
                return self.ddpg_critic_loss(predicted, label)
            case "behavior_clone":
                assert label is not None, "Behavior cloning loss requires label tensor"
                return self.behavior_cloning_loss(predicted, label)
            case "sac_actor":
                q = predicted
                q_other = kwargs["q_other"]
                log_prob = kwargs["log_prob"]
                log_alpha = kwargs["log_alpha"]
                return self.sac_actor_loss(q, q_other, log_prob, log_alpha)
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

    # TODO kwargs
    def sac_actor_loss(self, predicted_q: Tensor, predicted_q_other: Tensor, log_prob: Tensor, log_alpha: Tensor) -> Tensor:
        entropy = -log_prob

        return mean(-log_alpha.exp() * entropy - torch.min(predicted_q, predicted_q_other))

# endregion
