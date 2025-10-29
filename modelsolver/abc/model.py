from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from numpy import log
from torch import Tensor, no_grad, tensor
from torch.distributions import Normal
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

#region

# TODO 为了兼容 SAC, 强制所有模型必须实现 输出 mean 和 std, 然后内置采样
class IActor(ABC, Module):
    """策略网络接口"""

    if TYPE_CHECKING:
        def __call__(self, state: Tensor) -> tuple[Tensor, ...]:
            """前向传播

            Args:
                state (Tensor): 状态 s

            Returns:
                动作&其他参数 (Tensor): 动作 a 和其他参数 (比如 SAC 的 Actor 还会输出 logprob)
            """
            ...

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _action_mean(self, x: Tensor) -> Tensor:
        """计算动作的均值"""
        ...

    @abstractmethod
    def _action_std(self, x: Tensor) -> Tensor:
        """计算动作的标准差

        **注意**: 标准差必须是正数
        """
        ...

    @abstractmethod
    def _forward(self, state: Tensor) -> Tensor:
        """前向传播, 输入状态 s, 输出动作的特征, 将由此特征输入 `_action_mean` 和 `_action_std` 计算动作的均值和标准差.
        """
        ...

    def forward(self, state: Tensor) -> tuple[Tensor, ...]:
        """前向传播, 输入状态 s, 输出动作 a, 以及动作的对数概率 log_prob"""
        x = self._forward(state)
        action_mean = self._action_mean(x)
        action_std = self._action_std(x)
        return self._action_sample(action_mean, action_std)

    def _action_sample(self, action_mean: Tensor, action_std: Tensor) -> tuple[Tensor, Tensor]:
        """从动作分布中采样动作, 并计算该动作的对数概率

        + 动作值域在 [-1, 1]
        """
        dist = Normal(action_mean, action_std)
        sampled_action = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(sampled_action)

        action = torch.tanh(sampled_action)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # action = action * self.action_bound
        return action, log_prob


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
    + 已经实现的算法: DDPG, BC, SAC
    +
    """

    if TYPE_CHECKING:
        def __call__(self,
                     state: Tensor,
                     action: Tensor | None = None,
                     output: Literal["action",
                                     "target_action",
                                     "q", "q_other",
                                     "target_q",
                                     "target_q_sac"] = "action", ** kwargs) -> Tensor:
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

    def __init__(self,
                 actor: IActor,
                 critic: ICritic,
                 target_actor: IActor,
                 target_critic: ICritic,
                 # SAC, TD3 等需要多个 critic 的算法可以使用下面的参数
                 other_critic: ICritic,
                 other_target_critic: ICritic,
                 config: Any,

                 ):
        assert is_dataclass(config), "config 必须是 dataclass 类型"
        super().__init__()
        self._config = config
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.other_critic = other_critic
        self.other_target_critic = other_target_critic

        # 复制参数
        if actor is not None and critic is not None:
            self.target_actor.load_state_dict(actor.state_dict())
            self.target_critic.load_state_dict(critic.state_dict())
            self.other_critic.load_state_dict(other_critic.state_dict())

        # SAC 需要的 参数
        self.log_alpha = tensor(log(0.01), requires_grad=True, dtype=torch.float32)
        self.target_entropy = -config.action_channels  # type: ignore

    @property
    def config(self) -> Any:
        return self._config

    @no_grad()
    def soft_update_target_net(self, *targets: Literal["actor", "critic", "critic_other"], tau: float = 0.05):
        """软更新 target 网络

        + 经典 A-C: C有C', A~没有~A'
        + DDPG: C有C', A有A'
        Args:
            tau (float, optional): 软更新系数. Defaults to 0.005.
        """
        for target in targets:
            match target:
                case "actor":
                    pairs = zip(self.target_actor.parameters(), self.actor.parameters())
                case "critic":
                    pairs = zip(self.target_critic.parameters(), self.critic.parameters())
                case "critic_other":
                    pairs = zip(self.other_target_critic.parameters(), self.other_critic.parameters())
                case _:
                    raise ValueError("target must be 'actor' or 'critic'")

            for param_target, param in pairs:
                param_target.mul_(1.0 - tau)
                param_target.add_(param.data, alpha=tau)

    def forward(self,
                states: Tensor, actions: Tensor | None = None,
                output: Literal["action", "target_action", "q", "target_q", "target_q_sac"] = "action") -> Tensor:
        match output:
            case "action":
                return self._compute_action(states)
            case "target_action":
                return self._compute_action_with_target_net(states)
            case "q":
                return self._compute_q(states, actions)
            case "q_other":
                return self._compute_q_other(states, actions)
            case "target_q":
                return self._compute_q_with_target_net(states, actions)
            case "target_q_sac":
                return self._compute_q_with_target_net_by_sac(states, actions)
            case _:
                raise ValueError("output must be 'action', 'target_action', 'q' or 'target_q'")

    def _compute_action(self, states: Tensor) -> Tensor:
        """使用 actor 计算动作 a"""
        action, *_ = self.actor(states)
        return action

    def _compute_action_with_target_net(self, states: Tensor) -> Tensor:
        """使用 target actor 计算动作 a"""
        action, *_ = self.target_actor(states)
        return action

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

    def _compute_q_other(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """使用其他 critic 计算 q 值. 若 actions 为空, 则使用 actor 计算动作"""

        actions = actions if actions is not None else self._compute_action(states)
        q = self.other_critic(states, actions)
        return q

    def _compute_q_with_target_net_by_sac(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """使用 critic 计算 q 值. 若 actions 为空, 则使用 actor 计算动作. 该方法适用于 SAC 算法"""

        actions, log_probs = actions if actions is not None else self.actor(states)

        entropy = -log_probs
        target_q_1 = self.critic(states, actions)
        target_q_2 = self.other_critic(states, actions)

        return torch.min(target_q_1, target_q_2) + torch.exp(self.log_alpha) * entropy

# endregion
