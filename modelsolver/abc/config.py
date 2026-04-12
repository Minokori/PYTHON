"""配置模块, 定义了各种配置类, 包括数据配置、超参数配置、强化学习相关配置等"""
# region imports
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dataclasses_json import dataclass_json
from numpy import exp2, floor, log2


# endregion

@dataclass_json
@dataclass
class IConfig:
    if TYPE_CHECKING:
        def to_json(self, /, ensure_ascii=False, indent=4) -> str:
            """以 JSON 格式返回对象的字符串表示"""
            ...


# TODO agent hyper config
@dataclass_json
@dataclass
class HyperParameterConfig(IConfig):
    """超参数配置"""
    learning_rate: float = 1e-3
    """学习率, 也是 log_alpha 的学习率"""
    betas: tuple[float, float] = (0.9, 0.999)
    """Adam/AdamW 等优化器的 beta 参数"""
    weight_decay: float = 0.01
    """Adam/AdamW 等权重衰减"""
    eps: float = 1e-8
    """Adam/AdamW 等数值稳定性参数"""
    milestones: list[int] = field(default_factory=lambda: [100, 150, 200])
    """学习率调度器的里程碑"""
    gamma: float = 0.1
    """学习率调度器的衰减系数"""
    gamma_rl: float = 0.98
    """RL 的 奖励衰减系数"""
    epoch: int = 500
    """训练的总轮数"""

    actor_lr: float = 3e-4
    """Actor 的学习率"""
    critic_lr: float = 3e-3
    """Critic 的学习率"""
    policy_delay: int = 5
    """策略网络更新延迟系数"""
    batch_size: int = 8
    """每个批次的大小"""





# TODO 改成 IDataConfig, 目前这里面的配置项并不通用
@dataclass_json
@dataclass
class DataConfig:
    ratio: tuple[float, float, float] = (0.6, 0.2, 0.2)
    batch_size: int = 8
    k: int = 5
    def __post_init__(self):
        # 这里实现一些参数有效性校验
        t = log2(self.batch_size)
        t_int = floor(t)
        if t_int < t:
            logging.warning(f"批量大小 {self.batch_size} 不是 2 的指数倍, 考虑将其设置为 {exp2(t_int)} 或 {exp2(t_int + 1)}")


# region 强化学习相关配置类
@dataclass_json
@dataclass
class ReplayBufferConfig:
    capacity: int
    """经验回放缓冲区的容量"""
    state_dim: int
    """状态空间维度. 状态的形状为 (state_dim,)"""
    action_dim: int
    """动作空间维度. 动作的形状为 (action_dim,)"""
    minimal_capacity: int
    """在开始采样之前, 经验回放池中至少需要存储的序列数"""
    batch_size: int
    """每个采样批次的大小"""



@dataclass_json
@dataclass
class AgentHyperParameterConfig(HyperParameterConfig):
    """超参数配置"""
    gamma_rl: float = 0.98
    """RL 的 奖励衰减系数"""
    actor_lr: float = 3e-4
    """Actor 的学习率"""
    critic_lr: float = 3e-3
    """Critic 的学习率"""


@dataclass_json
@dataclass
class AgentConfig:
    state_channels: int
    action_channels: int
    hidden_channels: int
    target_entropy: float
    """目标熵. 一般设置为 `-action_channels`"""
    alpha_learnable:bool = True
    """是否学习温度参数 alpha"""
    alpha:float = 0.01
    """温度参数 alpha 的初始值"""


@dataclass_json
@dataclass
class EnvironmentConfig(IConfig):
    """环境配置"""

    terminated_delta: int = -1
    """是否启用终止状态检测.设置为 >0 的值时, 当连续若干时间步达到数值状态时, 环境将进入终止状态."""
    truncated_time: int = -1
    """时间步截断. 设置为 <0 则不启用截断."""

# endregion
