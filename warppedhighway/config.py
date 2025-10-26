from dataclasses import dataclass, field

from dataclasses_json import dataclass_json
from highway_env.road.lane import LineType
from numpy import float32
from numpy.typing import NDArray


# region 定义路网使用的模型
@dataclass_json
@dataclass
class StraightLaneModel:
    """直线车道"""
    from_node: str
    """起始点, 用于连接路段"""
    to_node: str
    """终点, 用于连接路段"""
    start_position: NDArray[float32]
    """起始位置 (x,y)"""
    end_position: NDArray[float32]
    """终点位置 (x,y)"""
    width: float = 3.75
    """车道宽度"""
    line_types: tuple[LineType, LineType] = (LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE)  # type: ignore
    """车道线类型, (左侧, 右侧), 0: 无, 1: 虚线, 2: 连续, 3: 连续线"""


@dataclass_json
@dataclass
class GoalModel:
    """车辆目标"""
    position: NDArray[float32]
    """位置 (x,y)"""
    heading: float
    """朝向, (东0, 逆时针, 弧度制)"""
    speed: float = 0.0
    """速度"""


@dataclass_json
@dataclass
class VehicleModel:
    """车辆"""
    start_position: NDArray[float32]
    """起始位置 (x,y)"""
    start_heading: float
    """起始朝向, (东0, 逆时针, 弧度制)"""
    start_speed: float
    length: float
    """长度"""
    width: float
    """宽度"""
    color: tuple[int, int, int]
    is_ego: bool
    """是否为智能体控制的车辆"""
    goal: GoalModel | None = None
    """车辆目标点"""


@dataclass_json
@dataclass
class ObstacleModel:
    """障碍物"""
    length: float
    """长度"""
    width: float
    """宽度"""
    position: NDArray[float32]
    """位置 (x,y)"""
    heading: float
    """朝向, (东0, 逆时针, 弧度制)"""
    diagonal: float = .0
    """对角线长度"""

    def __post_init__(self):
        self.diagonal = (self.length**2 + self.width**2)**0.5


@dataclass_json
@dataclass
class RoadNetworkModel:
    """路网"""
    lanes: list[StraightLaneModel] = field(default_factory=list)
    """车道"""
    vehicles: list[VehicleModel] = field(default_factory=list)
    """车辆"""
    obstacles: list[ObstacleModel] = field(default_factory=list)
    """障碍物"""
# endregion# endregion
