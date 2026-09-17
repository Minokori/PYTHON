"""连续高速公路的一些实用函数。主要用于转换动作/状态/观测的形状和标准化/反标准化"""
import torch


# region 与 highway_env 原生 Highway 环境保持兼容的配置示例
config = {"observation": {"type": "Kinematics"},
 "action": {"type": "DiscreteMetaAction"},
 "simulation_frequency": 15,
 "policy_frequency": 1,
 "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
 "screen_width": 600,
 "screen_height": 150,
 "centering_position": [0.3, 0.5],
 "scaling": 5.5,
 "show_trajectories": False,
 "render_agent": True,
 "offscreen_rendering": None,
 "manual_control": False,
 "real_time_rendering": False,
 "neighbour_vehicles_connected_lanes": True,
 "lanes_count": 4,
 "vehicles_count": 50,
 "controlled_vehicles": 1,
 "initial_lane_id": None,
 "duration": 40,
 "ego_spacing": 2,
 "vehicles_density": 1,
 "collision_reward": -1,
 "right_lane_reward": 0.1,
 "high_speed_reward": 0.4,
 "lane_change_reward": 0,
 "reward_speed_range": [20, 30],
 "normalize_reward": True,
 "offroad_terminal": False}
# endregion


# region 标准化 / 反标准化
FEATURE_ORDER = (
    "speed",
    "heading",
    "x",
    "y",
    "acceleration_x",
    "acceleration_y",
    "lane_id",
)
"""观测特征顺序, 与 raw_observation 的 7 个维度一一对应."""


def feature_ranges(normalization):
    """返回 7 个特征各自的 (low, high) 标准化范围.

    Args:
        normalization: FeatureNormalizationConfig 实例.

    Returns:
        list[tuple[float, float]]: 按 FEATURE_ORDER 排列的 7 个范围.
    """
    return [
        normalization.speed,
        normalization.heading,
        normalization.x,
        normalization.y,
        normalization.acceleration_x,
        normalization.acceleration_y,
        normalization.lane_id,
    ]


def _scale_to_unit(value, low, high):
    """将物理量线性映射到 [-1, 1]."""
    return (value - low) * (2.0 / (high - low)) - 1.0


def _scale_from_unit(value, low, high):
    """将 [-1, 1] 线性映射回物理量."""
    return (value + 1.0) * 0.5 * (high - low) + low


def normalize_state(states, normalization):
    """把物理状态标准化到 [-1, 1].

    heading 会先 wrap 到 [-pi, pi] 再按 heading_range 线性映射,
    从而正确处理角度周期性. 其余特征按各自 (low, high) 线性映射.
    若 normalization.clip 为 True, 结果会被裁剪到 [-1, 1].

    Args:
        states: shape (..., 7) 的物理状态张量, 特征顺序见 FEATURE_ORDER.
        normalization: FeatureNormalizationConfig.

    Returns:
        Tensor: shape 与 states 相同, dtype float32.
    """
    states = torch.as_tensor(states, dtype=torch.float32)
    out = states.clone()
    for index, (low, high) in enumerate(feature_ranges(normalization)):
        values = out[..., index]
        if index == 1:  # heading: 先处理角度周期性
            values = torch.atan2(torch.sin(values), torch.cos(values))
        values = _scale_to_unit(values, low, high)
        if normalization.clip:
            values = torch.clamp(values, -1.0, 1.0)
        out[..., index] = values
    return out


def denormalize_state(states, normalization):
    """把标准化状态反标准化回物理量.

    注意: 对 heading 的反标准化结果落在 [-pi, pi]; 对 lane_id
    会得到连续值, 需要调用方按 round 得到离散车道编号.

    Args:
        states: shape (..., 7) 的标准化状态张量.
        normalization: FeatureNormalizationConfig.

    Returns:
        Tensor: shape 与 states 相同, dtype float32.
    """
    states = torch.as_tensor(states, dtype=torch.float32)
    out = states.clone()
    for index, (low, high) in enumerate(feature_ranges(normalization)):
        out[..., index] = _scale_from_unit(out[..., index], low, high)
    return out
# endregion
