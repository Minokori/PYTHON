"""连续高速公路场景.

包括:
    - ContinuousHighwayEnvironmentConfig: 连续动作空间的高速公路环境配置类
    - ContinuousHighwayEnvironment: 连续动作空间的高速公路环境
    - HighwayReward: 连续高速公路场景的奖励函数.

"""

# region imports
from dataclasses import dataclass, field

import numpy as np
import torch
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from torch import Tensor

from modelsolver.abc.config import EnvironmentConfig
from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.reward import IReward


#endregion

# region 连续高速公路环境配置类
@dataclass
class ContinuousHighwayEnvironmentConfig(EnvironmentConfig):
    """连续动作空间的高速公路环境配置类, 继承自 EnvironmentConfig"""
    #region 场景定义
    road_length: float = 2000.0
    lane_count: int = 3
    lane_range:list[tuple[float,float]] = field(default_factory=lambda: [
        (3070.27, 3070.27+3.70),(3066.57, 3070.27),(3066.57-3.70, 3066.57)])
    #endregion

    #region 周车定义
    surrounding_vehicle_count: int = 15
    surrounding_vehicle_state: list[tuple[float,...]] = field(default_factory=lambda: [])
    #endregion

    # region 自车定义
    ego_vehicle_state: tuple[float,...] = field(default_factory=lambda: (20, 0.0, 10,3,068.42, 0,0,1))
    # endregion
    truncated_time = 400
# endregion

# action: 加速度, 前轮转向角.

# region 奖励函数
DISTANCETHRESHOLD:Tensor = torch.tensor(0.6827) #(1 / torch.sqrt(torch.tensor(2.0)))
"""距离阈值, 用于计算高斯函数的标准差, 68.27% 的积分在 [-1σ, 1σ] 区间内
"""
def sigma_from_roi(roi_meter: float) -> Tensor:
    """根据距离和累计积分求高斯函数标准差

    Args:
        roi_meter (float): 距离(单位:m)
    Returns:
        标准差 (Tensor): 标准差
    """
    return roi_meter / (torch.sqrt(torch.tensor(2.0)) * torch.erfinv(DISTANCETHRESHOLD))

def gaussian_weights(distance: Tensor, sigma: Tensor) -> Tensor:
    """根据距离和标准差生成权重

    Args:
        distance (np.ndarray): 各周车与 ego 的距离 (在本车前为正, 本车后为负), shape(B,15)
        sigma (float): 高斯函数标准差

    Returns:
        各周车的权重 (np.ndarray): shape(B,15)
    """


    # 计算权重
    weight = torch.exp(-(distance ** 2) / (2 * sigma ** 2))

    # 按批次归一化: 每个样本的 15 辆车权重之和为 1, (B,15) / (B,1)
    weight = weight / weight.sum(dim=-1, keepdim=True)

    # 避免除以 0 导致的 NaN / Inf (兜底)
    weight = torch.where(torch.isnan(weight) | torch.isinf(weight), 0, weight)
    return weight



class HighwayReward(IReward):
    """连续高速公路场景的奖励函数."""

    _B = Tensor([0,0, 9000, 3066.57-3.70, 0, 0, 0]) # [速度,航向角, x, y,  加速度x, 加速度y, lane] 的最小值
    _W = Tensor([1/(120/3.6), 1.0, 1/2000, 1/(3.7*3),  1/3.6, 1/3.6, 1/2]) # [速度, x, y, 航向角, 加速度x, 加速度y, lane] 的单位修正权重

    WEIGHT_AROUND = Tensor([0.5, 0.0, 1.0, 1.0, 0.05, 0.05, 1.0]) # [速度, 航向角, x, y, 加速度x, 加速度y, lane] 的权重
    WEIGHT_EGO = Tensor([0.5, 0.05, 1.0, 0.0, 0.05, 0.05, 1.0])
    GOAL = Tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5]]) # (1,7)
    @property
    def is_learnable(self) -> bool:
        return False

    def forward(self, **kwargs: Tensor) -> tuple[Tensor, Tensor]:
        # 有效的key :state, next_state
        if "state" in kwargs:
            state = kwargs["state"].reshape(-1, 16, 7)

            around = self._reward_around(state)
            ego = self._reward_ego(state)
            return (around, ego)

        else:
            raise ValueError("必须提供 state 或 next_state 参数")

    def _standardize(self, observations:Tensor)->Tensor:
        """对观测值进行标准化处理,

        (B, 16, 7) -> (B, 16, 7)

        Args:
            observations (np.ndarray): shape =  (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]

        Returns:
            standardized_observations (np.ndarray): shape =  (B, 16, 7), 标准化后的观测值
        """
        obs_ = torch.clone(observations)
        return (obs_ -self._B) * self._W

    def _reward_around(self, observations:Tensor)->Tensor:
        """根据周车和自车的相对状态定义奖励 (越高越好, 不一定是正值)

        Args:
            observations (np.ndarray): shape =  (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]

        Returns:
            rewards (float): 奖励(越高越好, 正值), (B,1)
        """
        obs_ =torch.clone(observations)

        # 解包数据
        ego = obs_[:, 0:1, :]  # 自车数据 (B,1,7)
        others = obs_[:, 1:, :]  # 周车数据 (B,15,7)
        postion_x = ego[:, 0, 2]  # 自车的 x 坐标 (B,)

        # 周车与自车的差值
        diff_x = others[:, :, 2] - postion_x.reshape(-1,1)  # 周围车与当前车的 x 坐标差 (前车为正, 后车为负值), (B,15)
        diff = others - ego  # 周围车与当前车的状态差 (B,15,7)

        # 计算周车的价值
        value_cars = (diff * self._W) @ self.WEIGHT_AROUND # (B,15)


        # 根据 diff_x 生成高斯权重 (已按样本归一化), (B,15)
        sigma = sigma_from_roi(roi_meter=80)  # 68.27% 的积分在 [-1σ, 1σ] 区间内
        W2 = gaussian_weights(diff_x, sigma)

        # 逐样本对 15 辆车的价值加权求和: (B,15) * (B,15) 沿最后一维求和 -> (B,1)
        value = (value_cars * W2).sum(dim=-1, keepdim=True)

        return value

    def _reward_ego(self, observations:Tensor)->Tensor:
        """根据自车的状态定义奖励 (越高越好, 不一定是正值)

        Args:
            observations (np.ndarray): shape =  (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]

        Returns:
            rewards (float): 奖励(越高越好, 正值)
        """
        obs_ =torch.clone(observations)
        obs_ = self._standardize(obs_)  # 标准化

        # 解包数据
        ego = obs_[:, 0:1, :]  # 自车数据 (B,7)
        value_ego = -torch.abs(ego - self.GOAL) @ self.WEIGHT_EGO # (B,1)
        return value_ego

# endregion

# region 连续高速公路环境
class ContinuousHighwayEnvironment(IEnvironment,AbstractEnv):
    """连续动作空间的高速公路环境, 继承自 highway_env.AbstractEnv
    """
    # region classmethod and field
    @property
    def ZERO_ACTION(self) -> Tensor:
        return torch.zeros(2, dtype=torch.float32).cpu()
    @property
    def GOAL(self) -> Tensor:
        ego_goal= torch.tensor([
            120.0 / 3.6,  # goal speed
            0.0,  # goal heading
            11000.0, # goal x
            3070.27 + 1.85,  # goal y (middle of lane 0)
            0.0,  # goal acceleration_x
            0.0,  # goal acceleration_y
            1.0,  # goal lane (lane 0)
        ]).reshape(1, 7)

        other_goal = torch.zeros(15, 7, dtype=torch.float32)  # 不关心周车状态, 用0 填充
        return torch.cat([ego_goal, other_goal]).float().cpu()
    @classmethod
    def default_config(cls)->dict:
        config = super().default_config()
        utils.update_config(config, {
            "observation": {"type": "Kinematics"},
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-np.pi / 4, np.pi / 4],
                "longitudinal": True,
                "lateral": True,
                "dynamical": False,
                "clip": True,
            },
            "simulation_frequency": 50,
            "policy_frequency": 5,
            "duration": 400.0,
            "screen_width": 1200,
            "screen_height": 400,
            "scaling": 7.0,
            "centering_position": [0.5, 0.5],
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False,
            "neighbour_vehicles_connected_lanes": True,

        })
        return config
    # endregion

    def __init__(self, config:EnvironmentConfig, reward:IReward):
        self._config:ContinuousHighwayEnvironmentConfig = config # type: ignore
        self.ego_vehicle: Vehicle = None # type: ignore
        self.surrounding_vehicles = []
        self.reward_fn = reward
        self._step_count = 0
        self._last_observation: Tensor | None = None
        AbstractEnv.__init__(self, config=ContinuousHighwayEnvironment.default_config(), render_mode="human")




    # region properties, easy access way.
    @property
    def observation(self) -> Tensor:
        """当前的观察, (1自车+15周车)*7维状态, 展平为 (1, 112)"""
        # ego
        heading = float(self.ego_vehicle.heading)
        acc = float(self.ego_vehicle.action.get("acceleration", 0.0)) # type: ignore
        ego_state = torch.tensor([
            float(self.ego_vehicle.speed),
            heading,
            float(self.ego_vehicle.position[0]),
            float(self.ego_vehicle.position[1]),
            float(acc * np.cos(heading)),
            float(acc * np.sin(heading)),
            float(self.ego_vehicle.lane_index[2]),
        ], dtype=torch.float32).reshape(1, 7)

        # surrounding
        surrounding_states = torch.zeros(15, 7, dtype=torch.float32)
        for i, vehicle in enumerate(self.surrounding_vehicles):
            heading = float(vehicle.heading)
            acc = float(vehicle.action.get("acceleration", 0.0)) # type: ignore
            surrounding_states[i] = torch.tensor([
                float(vehicle.speed),
                heading,
                float(vehicle.position[0]),
                float(vehicle.position[1]),
                float(acc * np.cos(heading)),
                float(acc * np.sin(heading)),
                float(vehicle.lane_index[2]),
            ], dtype=torch.float32)

        return torch.cat([ego_state, surrounding_states], dim=0)

    @property
    def terminated(self) -> Tensor:
        """终止标志, 1表示成功, 0表示未终止, -1表示失败"""
        ego = self.ego_vehicle
        if ego.crashed or not bool(ego.on_road):
            return torch.tensor([-1.0], dtype=torch.float32).cpu()
        elif ego.position[0] >= self._config.road_length:
            return torch.tensor([1.0], dtype=torch.float32).cpu()
        else:
            return torch.tensor([0.0], dtype=torch.float32).cpu()
    # endregion



    # 私有方法, 用于环境内部使用
    def _reset(self):
        self._make_road()
        self._make_vehicles()

    def _make_road(self):
        network = RoadNetwork()

        for lane_range in self._config.lane_range:
            y0, y1 = lane_range
            lane_center = 0.5 * (y0 + y1)
            width = y1 - y0
            lane = StraightLane(
                start=np.array([9000.0, lane_center]),
                end=np.array([9000.0+self._config.road_length, lane_center]),
                width=float(width),
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS), # type: ignore
                speed_limit=120/3.6,
            )
            network.add_lane("start", "end", lane)

        self.road = Road(
            network=network,
            np_random=self.np_random, # type: ignore
            record_history=self.config["show_trajectories"],
        )

        self.road.neighbour_vehicles_connected_lanes = True

    def _make_vehicles(self):

        self.controlled_vehicles = []
        self.surrounding_vehicles:list[Vehicle] = []

        # 自车
        ego_cls = self.action_type.vehicle_class
        v, h, x,y, ax, ay, l = self._config.ego_vehicle_state

        self.ego_vehicle:Vehicle = ego_cls(
            road=self.road,
            position=(x,y),
            heading=h,
            speed=v,
        )
        self.ego_vehicle.on_state_update()
        self.controlled_vehicles.append(self.ego_vehicle)
        self.road.vehicles.append(self.ego_vehicle)


        # 周车
        n = self._config.surrounding_vehicle_count


        if len(self._config.surrounding_vehicle_state) == n:
            for state in self._config.surrounding_vehicle_state:
                v, h, x,y, ax, ay, l = state

                lane = self.road.network.get_lane(("start", "end", int(l)))
                vehicle = IDMVehicle(
                    road=self.road,
                    position=lane.position(float(x), 0.0),
                    heading=float(h),
                    speed=float(v),
                    target_speed=120/3.6,
                    route=[("start", "end", None)],# type: ignore
                )
                vehicle.on_state_update()
                vehicle.collidable = True
                self.surrounding_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
        else:
            xs = np.linspace(9100.0, 9100.0+1000.0, n)

            for i, x in enumerate(xs):
                lane_id = i % 3
                lane = self.road.network.get_lane(("start", "end", lane_id))
                speed = float(self.np_random.uniform(60/3.6, 120/3.6))
                vehicle = IDMVehicle(
                    road=self.road,
                    position=lane.position(float(x), 0.0),
                    heading=0.0,
                    speed=speed,
                    target_speed=120/3.6,
                    route=[("start", "end", None)],# type: ignore
                )
                vehicle.on_state_update()
                vehicle.collidable = True
                self.surrounding_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)

    def _simulate(self, action=None):
        frames = int(
            self.config["simulation_frequency"]
            // self.config["policy_frequency"]
        )

        if action is not None:
            self.action_type.act(action)

        for frame in range(frames):
            self.road.act()

            self.road.step(1.0 / self.config["simulation_frequency"])
            self.steps += 1

            if frame < frames - 1:
                self._automatic_rendering()

        self.enable_auto_render = False

    # Required by AbstractEnv; reward/termination are owned by the adapter.
    def _reward(self, action):
        return 0.0

    def _is_terminated(self):
        return bool(self.ego_vehicle and self.ego_vehicle.crashed)

    def _is_truncated(self):
        return self.time >= self.config["duration"]

    def _info(self, obs, action=None):
        return {
            "speed": float(self.vehicle.speed),
            "crashed": bool(self.vehicle.crashed),
            "action": action,
        }
    def _set_surrounding_states(
        self,
        surrounding_observation: Tensor | np.ndarray,
    ) -> None:
        """根据输入设置周车的状态."""
        # check
        if isinstance(surrounding_observation, Tensor):
            surrounding_observation = surrounding_observation.detach().cpu().numpy()

        if surrounding_observation.shape != (15, 7):
            raise ValueError(f"Expected (15,7), got {surrounding_observation.shape}")

        # 依次设定周车状态
        for vehicle, state in zip(self.surrounding_vehicles, surrounding_observation, strict=True):
            speed, heading, x, y,*_ = state.tolist()
            # 更新车辆状态
            vehicle.position = np.asarray([x, y], dtype=np.float64)
            vehicle.heading = float(heading)
            vehicle.speed = float(speed)
            vehicle.on_state_update()
    def reset(self) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        self._step_count = 0
        super(AbstractEnv).reset()
        obs = self.observation
        reward, terminated = self.reward_fn(state = obs)
        truncated = torch.tensor([0])
        info = {}
        self._last_observation = obs.clone()
        return obs.cpu(), reward.cpu(), terminated.cpu(), truncated.cpu(), info


    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:
        # 若手动控制周围车辆, 则解包.
        if action.shape == (16,7):
            action = action[0:1,0:2]
            other = action[1:,:]
            self._set_surrounding_states(other)
        else:
            action = action.flatten()

        # 执行动作, 更新环境状态
        state = self.observation.clone()

        next_obs, reward, terminated, truncated, info = super(AbstractEnv).step(action.cpu().numpy())
        self._step_count += 1

        # 观察
        next_obs = self.observation.clone()

        # 计算奖励
        reward, done = self.reward_fn(state = state, action=action, next_state=next_obs)


        info  = {}
        self._last_observation = next_obs
        return next_obs.cpu(), reward.cpu(), done, truncated,info

# endregion














