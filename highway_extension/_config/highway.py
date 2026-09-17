"""连续高速公路场景使用的配置类"""
# region imports
from dataclasses import dataclass, field
from math import pi

from modelsolver.abc.config import EnvironmentConfig


# endregion



@dataclass
class ContinuousHighwayEnvironmentConfig(EnvironmentConfig):
    """连续动作空间的高速公路环境配置类, 继承自 EnvironmentConfig"""

    # region 道路定义
    bias:tuple[float, float, float,float,float,float,float] = field(default_factory=lambda:(0,       0, 9000, 3062.87, 0, 0, 1))
    """道路偏置, 用于调整道路在坐标系中的位置, shape (7,).

    `标准化后的 obs*weight+bias` 将映射到真实坐标系中, 其中 obs 为标准化后的观测, weight 为道路权重, bias 为道路偏置.
    """
    weight:tuple[float, float, float,float,float,float,float] = field(default_factory=lambda:(120/3.6, 1, 2000, 3.7*3,   1, 1, 2))
    """道路权重, 用于调整道路在坐标系中的影响, shape (7,)."""
    lane_width: float = 3.7
    """车道宽度 (m)."""

    @property
    def length(self) -> float:
        """道路长度 (m)."""
        return self.weight[2]
    @property
    def speed_limit(self) -> float:
        """道路限速 (m/s)."""
        return self.weight[0]
    @property
    def lanes(self) -> list[float]:
        """车道中心线 y 坐标列表, 从下到上."""
        return [(i + 0.5) * self.lane_width for i in range(int(self.weight[3] / self.lane_width))]
    # endregion

    #region 主车定义
    ego:tuple[float, float, float,float,float,float,float] = field(default_factory=lambda:(0.7, 0.0, 0, 0.5, 0.0, 0.0, 0.5))
    """主车初始状态 (速度, 航向角, x, y, ax, ay, lane)"""

    surround_count:int = 15
    """周车数量, 0 表示不生成周车."""
    #endregion

    # 继承自 EnvironmentConfig 的截断时间, 与 duration 保持一致.
    truncated_time: int = 400