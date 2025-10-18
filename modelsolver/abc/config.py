import logging
from dataclasses import dataclass, field
from typing import Literal

from dataclasses_json import dataclass_json
from numpy import exp2, floor, log2


@dataclass
class HyperParameterConfig:
    """超参数配置"""
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8
    milestones: list[int] = field(default_factory=lambda: [100, 150, 200])
    gamma: float = 0.1
    epoch: int = 500


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
