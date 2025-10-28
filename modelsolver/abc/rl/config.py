from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ReplayBufferConfig:
    capacity: int
    """经验回放缓冲区的容量"""
    state_dim: int
    """状态空间维度. 状态的形状为 (state_dim,)"""
    action_dim: int
    """动作空间维度. 动作的形状为 (action_dim,)"""
    batch_size: int
    """每个采样批次的大小"""
