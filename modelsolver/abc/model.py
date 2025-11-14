from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from numpy import log
from torch import Tensor, no_grad
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

# region RL

class IActor(ABC, Module):
    """策略网络接口. 输入 s, 输出 a 和采样到 a 的 log_prob.

    输出的动作取值范围在 [-1, 1] 之间.

    ---
    *需要重载的方法:*
    + `_action_mean`
    + `_action_std`
    + `_forward`
    """

    if TYPE_CHECKING:
        def __call__(self, state: Tensor) -> tuple[Tensor, ...]:
            """前向传播

            Args:
                state (Tensor): 状态 s

            Returns:
                动作&log_prob (Tensor): 动作 a 和 log_prob
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
        action_mean, action_std = self._action_mean(x), self._action_std(x)
        action, log_prob = self._action_sample(action_mean, action_std)
        return action, log_prob

    def _action_sample(self, action_mean: Tensor, action_std: Tensor) -> tuple[Tensor, Tensor]:
        """从动作分布中采样动作, 并计算该动作的对数概率

        + 动作值域在 [-1, 1]
        """
        dist = Normal(action_mean, action_std)
        sampled_action = dist.rsample()

        log_prob = dist.log_prob(sampled_action)
        action = torch.tanh(sampled_action)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        # 把动作维度求和，变为 (batch, 1)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
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
                                     "action_with_log_prob",
                                     "q", "q_other",
                                     "target_q",
                                     "target_q_sac", "target_q_td3"] = "action", ** kwargs) -> Tensor:
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
        """策略网络"""
        self.critic = critic
        """值网络"""
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.other_critic = other_critic
        self.other_target_critic = other_target_critic

        # 复制参数
        self.init_target_nets()

        # SAC 需要的 参数
        self.log_alpha = torch.tensor(log(0.01), requires_grad=True, dtype=torch.float32)

    @property
    def config(self) -> Any:
        return self._config

    def init_target_nets(self):
        """target 网络的参数从 actor 和 critic 复制"""
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.other_target_critic.load_state_dict(self.other_critic.state_dict())

    @no_grad()
    def soft_update_target_net(self, *targets: Literal["actor", "critic", "critic_other"], tau: float = 0.005):
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

    def forward(self, states: Tensor, actions: Tensor | None = None,
                output: str = "action") -> Tensor | tuple[Tensor, ...]:
        match output:
            # Actions
            case "action":
                return self._compute_action(states)
            case "action_with_log_prob":
                return self._compute_action_log_prob(states)
            case "target_action":
                return self._compute_action_with_target_net(states)

            # Q values
            case "q":
                return self._compute_q(states, actions)
            case "q_other":
                return self._compute_q_other(states, actions)
            case "target_q":
                return self._compute_q_with_target_net(states, actions)
            case "target_q_sac":
                return self._compute_q_with_target_net_by_sac(states, actions)
            case "target_q_td3":
                return self._compute_q_with_target_net_by_td3(states, actions)
            case _:
                raise ValueError("output must be 'action', 'target_action', 'q' or 'target_q'")

    def _compute_action(self, states: Tensor) -> Tensor:
        """使用 actor 计算动作 a"""
        action, _ = self.actor(states)
        return action

    def _compute_action_log_prob(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """使用 actor 计算动作 a 及其对数概率 log_prob"""
        action, log_prob = self.actor(states)
        return action, log_prob

    def _compute_action_with_target_net(self, states: Tensor) -> Tensor:
        """使用 target actor 计算动作 a"""
        action, _ = self.target_actor(states)
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
        assert actions is None, "SAC 计算 target Q 时, 不允许传入 actions, 必须使用 actor 计算动作"
        actions, log_probs = self.actor(states)
        entropy = -log_probs
        target_q_1 = self.target_critic(states, actions)
        target_q_2 = self.other_target_critic(states, actions)

        return torch.min(target_q_1, target_q_2) + self.log_alpha.exp() * entropy

    def _compute_q_with_target_net_by_td3(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        """使用 critic 计算 q 值. 若 actions 为空, 则使用 actor 计算动作. 该方法适用于 SAC 算法"""
        assert actions is None, "SAC 计算 target Q 时, 不允许传入 actions, 必须使用 actor 计算动作"
        actions, log_probs = self.target_actor(states)
        target_q_1 = self.target_critic(states, actions)
        target_q_2 = self.other_target_critic(states, actions)

        return torch.min(target_q_1, target_q_2)
#endregion