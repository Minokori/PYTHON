# region imports
import logging
from dataclasses import dataclass, field
from typing import Literal

from dataclasses_json import dataclass_json
from numpy import exp2, floor, log2


# endregion

# TODO agent hyper config
@dataclass
class HyperParameterConfig:
    """超参数配置"""
    learning_rate: float = 1e-3
    """学习率"""
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






@dataclass_json
@dataclass
class DataConfig:
    pickle_file_path: str
    sample_columns: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    ratio: tuple[float, float, float] = (0.6, 0.2, 0.2)
    batch_size: int = 8
    k: int = 5
    chunk_size: int = 0
    """一条序列裁剪到的长度, 一条序列可能因此裁剪为若干条数据
    则不进行裁剪
    """

    chunk_num: int = 1
    """一条序列裁剪到的个数.
    仅当 chunk_mode 为 "random" 时有效.
    """
    chunk_mode: Literal["random", "sequential"] = "sequential"

    def __post_init__(self):
        # 这里实现一些参数有效性校验
        t = log2(self.batch_size)
        t_int = floor(t)
        if t_int < t:
            logging.warning(f"批量大小 {self.batch_size} 不是 2 的指数倍, 考虑将其设置为 {exp2(t_int)} 或 {exp2(t_int + 1)}")


# region RL
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
class AgentHyperParameterConfig():
    """超参数配置"""
    simple: HyperParameterConfig = field(default_factory=lambda: HyperParameterConfig())
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

# endregion RL
