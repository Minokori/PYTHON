"""连续高速公路场景使用的配置类
"""

# region imports
from dataclasses import dataclass, field


from modelsolver.abc.config import EnvironmentConfig


#endregion

@dataclass
class ContinuousHighwayEnvironmentConfig(EnvironmentConfig):
    """连续动作空间的高速公路环境配置类, 继承自 EnvironmentConfig"""
    # 场景定义
    road_length: float = 2000.0
    """路段长度"""
    lane_count: int = 3
    """车道数量"""
    lane_range:list[tuple[float,float]] = field(default_factory=lambda: [
        (3070.27, 3070.27+3.70),(3066.57, 3070.27),(3066.57-3.70, 3066.57)])
    """车道范围,

    [(y0,y1), (y0,y1), (y0,y1)] 3个车道的纵向范围, 单位:m
    """

    # 周车定义
    surrounding_vehicle_count: int = 15
    """周车数量"""
    surrounding_vehicle_state: list[tuple[float,...]] = field(default_factory=lambda: [])
    """周车状态列表, 每个元素为一个元组, 包含 7 个元素:

    (速度, 航向角(rad), x, y, 加速度x, 加速度y, lane)

    *若为空列表, 则随机生成周车状态*"""

    # 自车定义
    ego_vehicle_state: tuple[float,...] = field(default_factory=lambda: (20, 0.0, 9010,3068.42, 0,0,1))
    """自车状态, 包含 7 个元素:

    (速度, 航向角(rad), x, y, 加速度x, 加速度y, lane)
    """
    # endregion
    truncated_time = 400
    """超时时间(秒).
    """
