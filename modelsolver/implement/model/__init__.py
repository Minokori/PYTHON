from torch import Tensor

from modelsolver.abc.config import AgentConfig
from modelsolver.abc.model import IActor, IAgentModel, ICritic


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


class DefaultAgent(IAgentModel):
    """默认智能体模型, 适用于大多数算法.

    配置类为 `modelsolver.abc.config.AgentConfig`.
    """

    def __init__(
            self,
            actor: IActor,
            critic: ICritic,
            target_actor: IActor,
            target_critic: ICritic,
            other_critic: ICritic,
            other_target_critic: ICritic,
            config: AgentConfig):
        super().__init__(actor, critic, target_actor, target_critic, other_critic, other_target_critic, config)

    @property
    def config(self) -> AgentConfig:
        return self._config  # type: ignore

    @property
    def name_for_save(self) -> str:
        return "DefaultAgent"
