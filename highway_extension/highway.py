"""连续高速公路场景.

包括:
    - ContinuousHighwayEnvironmentConfig: 连续动作空间的高速公路环境配置类
    - ContinuousHighwayEnvironment: 连续动作空间的高速公路环境
    - HighwayReward: 连续高速公路场景的奖励函数.

"""

# region imports

import numpy as np
import torch
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle
from torch import Tensor, tensor

from highway_extension._config.highway import \
    ContinuousHighwayEnvironmentConfig
from highway_extension._reward.highway import HighwayReward
from highway_extension._utils.highway import config as default_config
from modelsolver.abc.config import EnvironmentConfig
from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.reward import IReward


#endregion

# pylint: disable=R0902, C0321
__all__ = [
    "ContinuousHighwayEnvironmentConfig",
    "ContinuousHighwayEnvironment",
    "HighwayReward",]

class ContinuousHighwayEnvironment(IEnvironment,HighwayEnv):
    """连续动作空间的高速公路环境, 继承自 highway_env.AbstractEnv
    """
    # region properties
    @property
    def ZERO_ACTION(self) -> Tensor:
        return torch.zeros(2, dtype=torch.float32).cpu()
    @property
    def GOAL(self) -> Tensor:
        # presence, x, y, vx, vy, cos(heading), sin(heading)
        # 目标状态: 车道中间, 速度为 0.5*max_speed, heading = 0
        ego_goal = Tensor([1.0, # presence
                           1.0, # x (绝对)
                           0.5, # y (绝对), 环境归一化方法: y真实/车道*默认车道宽(4m)
                           120/3.6/80, # vx (绝对, 默认的归一化为 -40~40 m/s, 归一化后为 -1~1)
                           0.0, # vy (绝对)
                           1.0, # cos(heading)
                           0.0  # sin(heading)
                           ]).reshape(1, 7) # shape = (1, 7)
        return ego_goal.float().flatten().cpu() # shape = (7,)


    # endregion

    def __init__(self, config:EnvironmentConfig, reward:IReward):
        self._config:ContinuousHighwayEnvironmentConfig = config # type: ignore
        self._ego_vehicle: Vehicle = None # type: ignore
        self._surrounding_vehicles = []
        self._reward_fn = reward
        self._passed = 0
        self._passed_vehicle = 0
        HighwayEnv.__init__(self, config=default_config, render_mode="human")

    # region properties, easy access way.
    @property
    def _get_observation(self) -> Tensor:
        """使用 highway_env 的 Kinematics 观测. 输出 shape = (1, vehicles_count * len(features)).

        该观测包含自车 + 最近若干辆周车的 relative x/y/vx/vy 等特征,
        比“按绝对 x 距离取最近 3 辆车”更适合避撞和车道决策.
        """
        obs = np.nan_to_num(np.asarray(self.observation_type.observe(), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if not self._config.with_goal:
            return torch.from_numpy(obs).flatten().unsqueeze(0).cpu()
        else:
            obs_with_goal = np.concatenate([obs, self.GOAL], axis=0)
            return torch.from_numpy(obs_with_goal).flatten().unsqueeze(0).cpu()
    @property
    def terminated(self) -> Tensor:
        """终止标志, 1表示成功, 0表示未终止, -1表示失败"""
        ego = self._ego_vehicle
        if ego.crashed or not bool(ego.on_road):  # 碰撞
            return torch.tensor([-1.0], dtype=torch.float32).cpu()
        elif ego.position[0] >= self._config.length:  # 到达终点
            return torch.tensor([1.0], dtype=torch.float32).cpu()
        elif self._ego_vehicle.velocity[0] <= 60/3.6:  # 速度过低
            return torch.tensor([-1.0], dtype=torch.float32).cpu()
        else:# 正常行驶
            return torch.tensor([0.0], dtype=torch.float32).cpu()
    # endregion



    # region 私有方法, 用于环境内部使用
    def _reset(self):
        self._make_road()
        self._make_ego_vehicle()
        self._make_surrounding_vehicles()
        self._make_obstacles()

    def _make_road(self):
        network = RoadNetwork()
        length = self._config.length
        lane_width = self._config.lane_width
        speed_limit = self._config.speed_limit
        # 创建车道
        for lane_center_y in self._config.lanes:
            lane = StraightLane(
                start=np.array([0, lane_center_y]),
                end=np.array([0+length, lane_center_y]),
                width=float(lane_width),
                line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS), # type: ignore
                speed_limit=speed_limit,
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
        v, h, x,y, ax, ay, l = (np.array(self._config.ego) * self._config.weight).tolist()
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

        speeds = np.random.uniform(low = 60/3.6, high=100/3.6, size=self._config.surround_count)
        target_speeds = np.random.uniform(low = 60/3.6, high=90/3.6, size=self._config.surround_count)


        for i in range(self._config.surround_count):
            vehicle:IDMVehicle = IDMVehicle.create_random(road=self.road,lane_from="start", lane_to="end", spacing=1, speed=speeds[i])  # type: ignore
            vehicle.target_speed = target_speeds[i]
            vehicle.route = [("start", "end", None)]  # type: ignore
            vehicle.on_state_update()
            self._surrounding_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
    def _make_obstacles(self):
        """创建障碍物作为路标"""
        xs = np.arange(0, self._config.length, 50)
        for x in xs:
            box1 = Obstacle(self.road, position=(x, -2), heading=0)
            self.road.objects.append(box1)

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
        HighwayEnv.reset(self)
        # obs = self.observation
        # reward, terminated = self._reward_fn(state = obs)
        self._passed = 0
        self._passed_vehicle = 0
        truncated = torch.tensor([0])
        info = {}
        return self._get_observation.cpu(), tensor(0.0), tensor(0.0), truncated.cpu(), info


    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:
        # 若手动控制周围车辆, 则解包.
        if action.shape == (16,7):
            other = action[1:,:]
            action = action[0,0:2]

            self._set_surrounding_states(other)
        else:
            action = action.flatten()

        # 手动将方向盘的角度限制在一个较小的角度内
        action[1] *= 0.1

        prev_phi = self._lane_gap_potential()
        _, reward, _, truncated, _ = HighwayEnv.step(self, action.cpu().numpy())

        # 从 highway_env 继承的默认奖励是:
        #   reward = collision_reward*crashed + high_speed_reward*clip(forward_speed, reward_speed_range)
        #   reward *= on_road
        # 因此 _utils.highway.config 中关闭了 normalize_reward,
        # 否则停车时也会得到约 0.71 的奖励, 智能体就会学会“停车等”.
        if not isinstance(reward, float):
            reward = float(reward)

        # 1) 车道保持: 对横向偏移做稠密惩罚, 让车保持在车道中心附近.
        lat = float(self._ego_vehicle.lane_offset[1])
        lane_width = float(self._config.lane_width)
        reward -= 2.0 * min((lat / lane_width) ** 2, 4.0)

        # 2) 安全跟车: 离前车太近或接近过快时给强惩罚.
        forward_speed = max(float(self._ego_vehicle.speed * np.cos(self._ego_vehicle.heading)), 0.0)
        front_gap, front_ttc = self._front_risk()
        safe_gap = 8.0 + 1.5 * forward_speed
        if front_gap < safe_gap:
            ratio = front_gap / safe_gap
            reward -= 2.5 * (1.0 - ratio) ** 2
        if front_ttc < 2.0:
            reward -= 3.0 * (1.0 - front_ttc / 2.0) ** 2

        # 2b) 对任何过近车辆(前/侧)进行连续避撞惩罚.
        reward -= 0.6 * self._nearby_vehicle_penalty()

        # 2c) 车道通畅势能 shaping: 鼓励移动到前方空间更大的车道.
        next_phi = self._lane_gap_potential()
        reward += 5.0 * (0.98 * next_phi - prev_phi)

        # 2d) 超车事件奖励: 每新超过一辆车给一次奖励.
        behind = sum(1 for v in self._surrounding_vehicles
                     if float(v.position[0]) < float(self._ego_vehicle.position[0]))
        if behind > self._passed_vehicle:
            reward += 10.0 * (behind - self._passed_vehicle)
            self._passed_vehicle = behind

        # 3) 事件奖励/惩罚.
        if self.terminated.item() > 0:
            reward += 100.0
        elif self.terminated.item() < 0:
            reward -= 50.0

        # 4) 数值保护: 防止 inf/inf 等意外值污染 replay buffer 和梯度.
        if not np.isfinite(reward):
            reward = 0.0

        return self._get_observation.cpu(), tensor(reward).cpu(), self.terminated.cpu(), truncated, {}

    def _front_clearance(self) -> float:
        """返回同车道正前方最近车辆的纵向距离, 若无前车返回 inf."""
        ego = self._ego_vehicle
        if ego is None:
            return float("inf")
        min_gap = float("inf")
        for vehicle in self._surrounding_vehicles:
            dx = float(vehicle.position[0] - ego.position[0])
            if dx <= 0:
                continue
            dy = abs(float(vehicle.position[1] - ego.position[1]))
            if dy < self._config.lane_width:
                min_gap = min(min_gap, dx)
        return min_gap

    def _front_risk(self) -> tuple[float, float]:
        """返回同车道正前方最近车辆的 (间距, TTC). 无前车时返回 (inf, inf)."""
        ego = self._ego_vehicle
        if ego is None:
            return float("inf"), float("inf")
        front_gap = float("inf")
        lead_speed = 0.0
        for vehicle in self._surrounding_vehicles:
            dx = float(vehicle.position[0] - ego.position[0])
            if dx <= 0:
                continue
            dy = abs(float(vehicle.position[1] - ego.position[1]))
            if dy < self._config.lane_width and dx < front_gap:
                front_gap = dx
                lead_speed = float(vehicle.speed)
        if not np.isfinite(front_gap):
            return front_gap, float("inf")
        forward_speed = max(float(ego.speed * np.cos(ego.heading)), 0.0)
        closing_speed = max(forward_speed - lead_speed, 0.0)
        ttc = front_gap / max(closing_speed, 1e-3)
        return front_gap, ttc

    def _front_clearance_for_lane(self, lane_id: int) -> float:
        """返回指定车道正前方最近车辆的纵向距离, 若无前车返回 inf."""
        ego = self._ego_vehicle
        if ego is None:
            return float("inf")
        min_gap = float("inf")
        for vehicle in self._surrounding_vehicles:
            if int(vehicle.lane_index[2]) != int(lane_id):
                continue
            dx = float(vehicle.position[0] - ego.position[0])
            if dx > 0:
                min_gap = min(min_gap, dx)
        return min_gap

    def _lane_gap_potential(self) -> float:
        """以“所有车道中安全的前方间距”作为势能, 用于 potential-based shaping.

        只考虑大于当前安全跟车距离的车道间距, 避免奖励驶入同样不安全的小间隙.
        """
        ego = self._ego_vehicle
        if ego is None:
            return 0.0
        forward_speed = max(float(ego.speed * np.cos(ego.heading)), 0.0)
        safe_gap = 8.0 + 1.5 * forward_speed
        gaps = [self._front_clearance_for_lane(lane_id) for lane_id in range(len(self._config.lanes))]
        safe_gaps = [g for g in gaps if np.isfinite(g) and g > safe_gap]
        if not safe_gaps:
            return 0.0
        best_gap = max(safe_gaps)
        if not np.isfinite(best_gap):
            return 1.0
        return best_gap / (best_gap + 30.0)

    def _nearby_vehicle_penalty(self, radius: float = 20.0) -> float:
        """对过近的周车给出连续惩罚, 用于减少碰撞."""
        ego = self._ego_vehicle
        if ego is None:
            return 0.0
        penalty = 0.0
        for vehicle in self._surrounding_vehicles:
            dx = float(vehicle.position[0] - ego.position[0])
            if dx < -10.0:
                continue
            dy = float(vehicle.position[1] - ego.position[1])
            dist = float(np.hypot(dx, dy))
            if dist < radius:
                penalty += (1.0 - dist / radius)
        return min(penalty, 2.0)

    # region override 没有实际作用
    def _is_terminated(self):
        if not self._ego_vehicle:
            return False
        if self._ego_vehicle.crashed:
            return True
        if not self._ego_vehicle.on_road:
            return True
        return False

    def _is_truncated(self): return self.time >= self._config.truncated_time

    # endregion






