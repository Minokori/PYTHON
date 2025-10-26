from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Literal

from modelsolver.abc.model import IModel
from torch import Tensor
from torch.nn import Module


class IActor(ABC, Module):
    """策略网络接口"""

    if TYPE_CHECKING:
        def __call__(self, state: Tensor) -> Tensor:
            """前向传播

            Args:
                state (Tensor): 状态 s

            Returns:
                动作 (Tensor): 动作 a
            """
            ...

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        """前向传播, 输入状态 s, 输出动作 a"""


class ICritic(ABC, Module):
    """值网络接口"""

    if TYPE_CHECKING:
        def __call__(self, state: Tensor, action: Tensor) -> Tensor:
            """前向传播

            Args:
                state (Tensor): 状态 s
                action (Tensor): 在状态 s 下, 策略网络(Actor)输出的动作 a

            Returns:
                动作的价值 (Tensor): 状态 s下, 动作的价值 Q(s,a)
            """
            ...

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        ...


class IAgentModel(IModel):
    """智能体模型接口.

    + 以 Actor-Critic 架构为基础的智能体模型
    """

    if TYPE_CHECKING:
        def __call__(self,
                     state: Tensor,
                     action: Tensor | None = None,
                     output: Literal["action",
                                     "target_action",
                                     "q",
                                     "target_q"] = "action", ** kwargs) -> Tensor:
            """
            + 输入 s, 返回 a (策略计算)
            + 输入 (s,a), 返回 q (值计算)
                + 计算 Q(s,a) 时, 若 action 为空, 则使用 Actor 计算动作. 使用的 Actor 由 output 决定, 若 output 为 "target_action", 则使用 target Actor, 否则使用当前 Actor.

            Args:
                state (Tensor): 状态 s
                action (Tensor | None, optional): 动作 a. Defaults to None.
                output (Literal[&quot;action&quot;, &quot;target_action&quot;, &quot;q&quot;, &quot;target_q&quot;], optional): 计算目标. Defaults to "action".

            Returns:
                期望Q或者动作A (Tensor): q 或者 a
            """
            ...

    def __init__(self, actor: IActor, critic: ICritic, config: Any):
        assert is_dataclass(config), "config 必须是 dataclass 类型"
        super().__init__()
        self.actor = actor
        self.critic = critic
        self._config = config
        if actor is not None and critic is not None:
            self.target_actor = deepcopy(actor)
            self.target_critic = deepcopy(critic)
            self.target_actor.load_state_dict(actor.state_dict())
            self.target_critic.load_state_dict(critic.state_dict())

    @property
    def config(self) -> Any:
        return self._config

    def soft_update(self, *targets: Literal["actor", "critic"], tau: float = 0.005):
        """软更新 target 网络

        + 经典 A-C: C有C', A~没有~A'
        + DDPG: C有C', A有A'
        Args:
            tau (float, optional): 软更新系数. Defaults to 0.005.
        """
        for target in targets:
            match target:
                case "actor":
                    target_net_params = self.target_actor.parameters()
                    net_params = self.actor.parameters()
                case "critic":
                    target_net_params = self.target_critic.parameters()
                    net_params = self.critic.parameters()
                case _:
                    raise ValueError("target must be 'actor' or 'critic'")

            for param_target, param in zip(target_net_params, net_params):
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    def forward(self,
                states: Tensor, actions: Tensor | None = None,
                output: Literal["action", "target_action", "q", "target_q"] = "action") -> Tensor:
        match output:
            case "action":
                return self._compute_action(states)
            case "target_action":
                return self._compute_action_with_target_net(states)
            case "q":
                return self._compute_q(states, actions)
            case "target_q":
                return self._compute_q_with_target_net(states, actions)
            case _:
                raise ValueError("output must be 'action', 'target_action', 'q' or 'target_q'")

    def _compute_action(self, states: Tensor) -> Tensor:
        """使用 actor 计算动作 a"""
        return self.actor(states)

    def _compute_action_with_target_net(self, states: Tensor) -> Tensor:
        """使用 target actor 计算动作 a"""
        return self.target_actor(states)

    def _compute_q_with_target_net(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """使用 target critic 计算 q 值, 若 actions 为空, 则使用 target actor 计算动作"""
        actions = actions if actions is not None else self._compute_action_with_target_net(states)
        q = self.target_critic(states, actions)
        return q

    def _compute_q(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """使用 critic 计算 q 值. 若 actions 为空, 则使用 actor 计算动作"""

        actions = actions if actions is not None else self._compute_action(states)
        q = self.critic(states, actions)
        return q
