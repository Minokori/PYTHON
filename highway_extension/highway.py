"""连续高速公路场景.

包括:
    - ContinuousHighwayEnvironmentConfig: 连续动作空间的高速公路环境配置类
    - ContinuousHighwayEnvironment: 连续动作空间的高速公路环境
    - HighwayReward: 连续高速公路场景的奖励函数.

"""

# region imports

import numpy as np
import torch
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from torch import Tensor

from highway_extension._config.highway import \
    ContinuousHighwayEnvironmentConfig
from highway_extension._reward.highway import HighwayReward
from modelsolver.abc.config import EnvironmentConfig
from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.reward import IReward


#endregion

# pylint: disable=R0902, C0321
__all__ = [
    "ContinuousHighwayEnvironmentConfig",
    "ContinuousHighwayEnvironment",
    "HighwayReward",]

class ContinuousHighwayEnvironment(IEnvironment,AbstractEnv):
    """连续动作空间的高速公路环境, 继承自 highway_env.AbstractEnv
    """
    # region classmethod and field
    _B = Tensor([0,0, 9000, 3066.57-3.70, 0, 0, 0])
    """
    标准化输入时用的偏移量

    [速度,航向角, x, y,  加速度x, 加速度y, lane] 的最小值
    """
    _W = Tensor([1/(120/3.6), 1.0, 1/2000, 1/(3.7*3),  1/3.6, 1/3.6, 1/2])
    """
    标准化输入时用的缩放因子

    [速度, x, y, 航向角, 加速度x, 加速度y, lane] 的单位修正权重
    """
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
            1.0,  # goal lane (lane 1)
        ]).reshape(1, 7)

        other_goal = torch.zeros(15, 7, dtype=torch.float32)  # 不关心周车状态, 用 0 填充
        return torch.cat([ego_goal, other_goal]).float().cpu()
    @property
    def STANDARDIZED_GOAL(self) -> Tensor:
        ego_goal= torch.tensor([
            1.0,  # goal speed
            0.0,  # goal heading
            1.0, # goal x
            0.5,  # goal y (middle of lane 0)
            0.0,  # goal acceleration_x
            0.0,  # goal acceleration_y
            0.5,  # goal lane (lane 1)
        ]).reshape(1, 7)

        other_goal = torch.zeros(15, 7, dtype=torch.float32)  # 不关心周车状态, 用 0 填充
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
        self._ego_vehicle: Vehicle = None # type: ignore
        self._surrounding_vehicles = []
        self._reward_fn = reward
        self._step_count = 0
        self._last_observation: Tensor | None = None
        AbstractEnv.__init__(self, config=ContinuousHighwayEnvironment.default_config(), render_mode="human")

    # region properties, easy access way.
    @property
    def observation(self) -> Tensor:
        """当前的观察+目标, shape = (1,224)

        状态和目标均为 (1自车+15周车)*7维状态, 展平为 (1, 112)
        """
        # ego
        heading = float(self._ego_vehicle.heading)
        acc = float(self._ego_vehicle.action.get("acceleration", 0.0)) # type: ignore
        ego_state = torch.tensor([
            float(self._ego_vehicle.speed),
            heading,
            float(self._ego_vehicle.position[0]),
            float(self._ego_vehicle.position[1]),
            float(acc * np.cos(heading)),
            float(acc * np.sin(heading)),
            float(self._ego_vehicle.lane_index[2]),
        ], dtype=torch.float32).reshape(1, 7)

        # surrounding
        surrounding_states = torch.zeros(15, 7, dtype=torch.float32)
        for i, vehicle in enumerate(self._surrounding_vehicles):
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

        obs = torch.cat([ego_state, surrounding_states], dim=0).reshape(1, 16, 7) # shape = (1, 16, 7)

        return torch.cat([obs, self.GOAL.reshape(1,16,7)], dim=2).reshape(1,-1).float().cpu() # shape = (1, 224)
    @property
    def standardized_observation(self) -> Tensor:
        """当前的标准化观察+目标, shape = (1,224)

        状态和目标均为 (1自车+15周车)*7维状态, 展平为 (1, 112)
        """
        # ego
        heading = float(self._ego_vehicle.heading)
        acc = float(self._ego_vehicle.action.get("acceleration", 0.0)) # type: ignore
        ego_state = torch.tensor([
            float(self._ego_vehicle.speed),
            heading,
            float(self._ego_vehicle.position[0]),
            float(self._ego_vehicle.position[1]),
            float(acc * np.cos(heading)),
            float(acc * np.sin(heading)),
            float(self._ego_vehicle.lane_index[2]),
        ], dtype=torch.float32).reshape(1, 7)

        # surrounding
        surrounding_states = torch.zeros(15, 7, dtype=torch.float32)
        for i, vehicle in enumerate(self._surrounding_vehicles):
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

        obs = torch.cat([ego_state, surrounding_states], dim=0).reshape(1, 16, 7) # shape = (1, 16, 7)
        obs = self._standardize(obs)

        return torch.cat([obs, self.STANDARDIZED_GOAL.reshape(1,16,7)], dim=2).reshape(1,-1).float().cpu() # shape = (1, 224)

    @property
    def terminated(self) -> Tensor:
        """终止标志, 1表示成功, 0表示未终止, -1表示失败"""
        ego = self._ego_vehicle
        if ego.crashed or not bool(ego.on_road):  # 碰撞
            return torch.tensor([-1.0], dtype=torch.float32).cpu()
        elif ego.position[0] >= self._config.road_length+9000.0:  # 到达终点
            return torch.tensor([1.0], dtype=torch.float32).cpu()
        else:  # 未终止
            return torch.tensor([0.0], dtype=torch.float32).cpu()
    # endregion



    # region 私有方法, 用于环境内部使用
    def _reset(self):
        self._make_road()
        self._make_ego_vehicle()
        self._make_surrounding_vehicles()

    def _make_road(self):
        network = RoadNetwork()

        # 创建车道
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

        # 创建道路
        self.road = Road(
            network=network,
            np_random=self.np_random, # type: ignore
            record_history=self.config["show_trajectories"],
        )

        # 设置道路属性
        self.road.neighbour_vehicles_connected_lanes = True

    def _make_ego_vehicle(self):
        self.controlled_vehicles = []
        ego_cls = self.action_type.vehicle_class
        v, h, x,y, ax, ay, l = self._config.ego_vehicle_state
        self._ego_vehicle:Vehicle = ego_cls(
            road=self.road,
            position=(x,y),
            heading=h,
            speed=v,
        )
        self._ego_vehicle.on_state_update()
        self.controlled_vehicles.append(self._ego_vehicle)
        self.road.vehicles.append(self._ego_vehicle)

    def _make_surrounding_vehicles(self):
        self._surrounding_vehicles:list[Vehicle] = []
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
                self._surrounding_vehicles.append(vehicle)
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
                self._surrounding_vehicles.append(vehicle)
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


    def _set_surrounding_states(
        self,
        surrounding_observation: Tensor,
    ) -> None:
        """根据输入设置周车的状态."""
        # TODO : 输入的周车状态应该是标准化的, 需要在这里进行反标准化处理, 以便设置车辆状态.
        if surrounding_observation.shape != (15, 7):
            raise ValueError(f"Expected (15,7), got {surrounding_observation.shape}")

        # 依次设定周车状态
        for vehicle, state in zip(self._surrounding_vehicles, surrounding_observation, strict=True):
            speed, heading, x, y,*_ = state.tolist()
            # 更新车辆状态
            vehicle.position = np.asarray([x, y], dtype=np.float64)
            vehicle.heading = float(heading)
            vehicle.speed = float(speed)
            vehicle.on_state_update()
    # endregion

    def reset(self) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        self._step_count = 0
        AbstractEnv.reset(self)
        obs = self.standardized_observation
        reward, terminated = self._reward_fn(state = obs)
        truncated = torch.tensor([0])
        info = {}
        self._last_observation = obs.clone()
        return self.standardized_observation.cpu(), reward.cpu(), terminated.cpu(), truncated.cpu(), info


    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:
        # 若手动控制周围车辆, 则解包.
        if action.shape == (16,7):
            other = action[1:,:]
            action = action[0,0:2]

            self._set_surrounding_states(other)
        else:
            action = action.flatten()

        # 执行动作, 更新环境状态
        state = self.standardized_observation.clone()

        # 手动将方向盘的角度限制在一个较小的角度内 (0.01 rad)
        action[1] *=0.01

        _, _, _, truncated, _ = AbstractEnv.step(self, action.cpu().numpy())
        self._step_count += 1

        # 观察
        next_obs = self.standardized_observation.clone()

        # 计算奖励
        reward, _ = self._reward_fn(state = state, action=action, next_state=next_obs)


        # 加入极高的惩罚, 如果车辆偏离车道太远, 或者车辆速度过低
        if self.terminated.item()<0:
            reward -= 20.0
        if self._ego_vehicle.speed < 60/3.6:
            reward -= 10.0


        self._last_observation = next_obs
        return  self.standardized_observation.cpu(), reward.cpu(), self.terminated.cpu(), truncated, {}

    # region override 没有实际作用
    def _reward(self, action): return 0.0

    def _is_terminated(self): return bool(self._ego_vehicle and self._ego_vehicle.crashed)

    def _is_truncated(self): return self.time >= self.config["duration"]

    def _info(self, obs, action=None): return {}

    def _standardize(self, observations:Tensor)->Tensor:
        """对观测值进行标准化处理

        (B, 16, 14) -> (B, 16, 7)

        Args:
            observations (np.ndarray): shape =  (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]

        Returns:
            standardized_observations (np.ndarray): shape =  (B, 16, 7), 标准化后的观测值
        """
        obs_ = torch.clone(observations)
        return (obs_ -self._B) * self._W
    # endregion














