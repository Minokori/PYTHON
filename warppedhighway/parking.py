"""包装后的 highway-env 停车环境"""


# region import
from typing import TYPE_CHECKING, TypedDict

from highway_env.envs import ParkingEnv
from highway_env.envs.common.observation import \
    KinematicsGoalObservation as _KinematicsGoalObservation
from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from numpy import float32, rad2deg, zeros_like
from numpy.linalg import norm
from numpy.typing import NDArray
from torch import Tensor, concatenate, from_numpy, tensor

from modelsolver.abc.environment import IEnvironment

from .config import (GoalModel, ObstacleModel, RoadNetworkModel,
                     StraightLaneModel, VehicleModel)


# endregion

# region 环境配置字典
class ActionType(TypedDict):
    type: str


class ObservationType(TypedDict):
    type: str
    features: list[str]
    scales: list[float]
    normalize: bool


class ParkingEnvironmentConfig(TypedDict):
    observation: ObservationType
    """观察空间配置"""
    action: ActionType
    """动作空间配置"""
    simulation_frequency: int
    """仿真频率"""
    policy_frequency: int
    """策略频率"""
    other_vehicles_type: str
    """其他车辆类型"""
    screen_width: int
    """屏幕宽度"""
    screen_height: int
    """屏幕高度"""
    centering_position: list[float]
    """屏幕中心位置"""
    scaling: float
    """缩放比例"""
    show_trajectories: bool
    """是否显示轨迹"""
    render_agent: bool
    """是否渲染智能体"""
    offscreen_rendering: bool
    """是否开启离屏渲染"""
    manual_control: bool
    """是否关闭手动控制车辆的位置. 设置为 `False` 后可以通过设置 steering 和 acceleration 来控制车辆的运动"""
    real_time_rendering: bool
    """是否开启实时渲染"""
    reward_weights: list[float]
    """奖励权重"""
    success_goal_reward: float
    """达到目标获得的奖励值"""
    collision_reward: float
    """碰撞奖励(设置为负数为惩罚)"""
    steering_range: float
    """转向范围, 弧度"""
    duration: int
    """每个回合的持续时间, 单位秒"""
    controlled_vehicles: int
    """智能体控制的车辆数量"""
    vehicles_count: int
    """环境中其他车辆的数量"""
    add_walls: bool
    """是否添加墙壁作为障碍物"""


class ObservationDict(TypedDict):
    observation: NDArray[float32]
    """观测. 和"""
    achieved_goal: NDArray[float32]
    desired_goal: NDArray[float32]


class KinematicsGoalObservation(_KinematicsGoalObservation):
    if TYPE_CHECKING:
        def observe(self) -> ObservationDict: ...
# endregion


class ParkingEnvironment(IEnvironment, ParkingEnv):
    """单车停车环境"""

    # region classmethod and field
    DEFAULT_CONFIG: ParkingEnvironmentConfig = {
        "observation": {
            "type": "KinematicsGoal",
            "features": [
                "x",
                "y",
                "vx",
                "vy",
                "cos_h",
                "sin_h"
            ],
            "scales": [
                1,
                1,
                1,
                1,
                1,
                1
            ],
            "normalize": True
        },
        "action": {
            "type": "ContinuousAction"
        },
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 800,
        "screen_height": 800,
        "centering_position": [
            0.5,
            0.5
        ],
        "scaling": 7,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
        "manual_control": False,
        "real_time_rendering": True,
        "reward_weights": [
            1.0,
            1.0,
            0.01,
            0.01,
            0.05,
            0.05
        ],
        "success_goal_reward": 0.12,
        "collision_reward": -10,
        "steering_range": 0.7853981633974483,
        "duration": 100,
        "controlled_vehicles": 1,
        "vehicles_count": 0,
        "add_walls": False
    }

    @classmethod
    def default_config(cls) -> ParkingEnvironmentConfig:
        return cls.DEFAULT_CONFIG.copy()

    if TYPE_CHECKING:
        observation_type_parking: KinematicsGoalObservation
        last_observation: ObservationDict
        """上一时刻的观测"""
        last_action: Tensor
        """上一时刻的动作"""
    # endregion

    def __init__(self, network: RoadNetworkModel = RoadNetworkModel()):
        self.network_model = network

    # region 配置路网结构

    def add_straight_lanes(self, *lane: StraightLaneModel):
        self.network_model.lanes.extend(lane)
        return self

    def add_vehicles(self, *vehicle_with_goal: tuple[VehicleModel, GoalModel | None]):
        for vehicle, goal in vehicle_with_goal:
            if goal:
                vehicle.goal = goal
            self.network_model.vehicles.append(vehicle)
        return self

    def add_obstacles(self, *obstacle: ObstacleModel):
        self.network_model.obstacles.extend(obstacle)
        return self

    def build_environment(self, config: ParkingEnvironmentConfig | None = None, render_mode: str = "human", ):
        c = ParkingEnvironment.DEFAULT_CONFIG
        if config:
            c.update(config)
        super().__init__(dict(c), render_mode)
        self.define_spaces()
        observation = self.observation_type.observe()
        self.last_observation = observation
        self.last_action = from_numpy(zeros_like(self.action_space.sample()))
        return self
    # endregion

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        """
        + 在时间步 t 环境处于状态 s_t，agent 根据观察 o_t 选动作 a_t；
        + 调用 env.step(a_t) 后，环境根据动力学转移到下一个状态 s_{t+1}，并返回与 s_{t+1} 对应的观测 o_{t+1}、奖励 r_t、终止标志 done 和 info。

        >>> next_ob, r, done, info = env.step(action)

        Args:
            action (NDArray[float32]): t 时刻的动作

        Returns:
            环境反馈 (tuple[NDArray[float32], float, bool, bool, dict]): t+1 时刻的观测、奖励、终止标志、截断标志和信息
        """
        # 记录上一个动作和观测
        last_action, last_observation = self.last_action, self.last_observation
        # 记录当前动作和观测
        action, observation = action, self.observation_type.observe()

        a = action.cpu().detach().numpy()
        # 执行动作
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(a)

        # 计算奖励等
        next_observation: ObservationDict = self.observation_type.observe()
        reward = self._reward(a, last_action.numpy(), observation, last_observation)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(next_observation, a)

        if self.render_mode == "human":
            self.render()

        # 更新 last_observation 和 last_action
        self.last_observation: ObservationDict = observation
        self.last_action = action

        return self._convert_observation(next_observation).float(), tensor(
            reward).float().reshape(1), tensor(terminated).float().reshape(1), truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        observation, info = super(ParkingEnv, self).reset(seed=seed, options=options)
        self.last_observation: ObservationDict = observation
        self.last_action = from_numpy(zeros_like(self.action_space.sample()))
        return self._convert_observation(observation).float(), tensor(0).float().reshape(1), tensor(0).float().reshape(1), False, info
    # endregion

    # region override private method

    def _reset(self):
        self._create_road()\
            ._create_vehicles()\
            ._create_obstacles()

    def _is_success(self, achieved_goal: NDArray[float32], desired_goal: NDArray[float32]) -> bool:

        # 真实世界尺度下的误差
        error = (achieved_goal - desired_goal) * self.config["observation"]["scales"]

        # 1 位置误差小于 0.25m2
        position_error = norm(error[:2], ord=2)

        # 2 速度误差小于 0.05m/s
        # speed_error = norm(error[2:4], ord=2)

        # 3 角度误差小于 5度
        heading = self.vehicle.heading
        goal_heading = self.vehicle.goal.heading  # type: ignore

        heading_error = abs(rad2deg(heading - goal_heading))

        return bool((position_error < 0.05) and (heading_error < 5))  # and (speed_error < 0.05)

    def _reward(
            self,
            action: NDArray[float32],
            last_action: NDArray[float32],
            observation: ObservationDict,
            last_observation: ObservationDict) -> float:
        # 1 累计奖励，根据当前的状态（x,y,vx,vy,cosh,sinh）和期望的状态计算的累计奖励 [-1,0]
        computed_reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], {})
        # 2 碰撞惩罚 [-5 | 0]
        collison_reward = self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        # 3 动作惩罚 (动作变化小有奖励，动作变化大惩罚) [-1,0]
        # action_reward = - norm(action - last_action, ord=2)
        # 4 位移奖励
        # move_reward = norm(last_observation["achieved_goal"][0:2] - last_observation["desired_goal"][0:2]) - norm(observation["achieved_goal"][0:2] - observation["desired_goal"][0:2]) / 2 - 0.5
        return computed_reward + collison_reward # + action_reward * 0.1 + move_reward * 0.1
    # endregion

    # region 创建路网

    def _create_road(self):
        """构建路网"""
        net = RoadNetwork()
        # add lanes
        for lane_model in self.network_model.lanes:
            net.add_lane(
                lane_model.from_node,
                lane_model.to_node,
                StraightLane(
                    lane_model.start_position,
                    lane_model.end_position,
                    lane_model.width,
                    line_types=lane_model.line_types,
                ),
            )

        # create road network
        self.road = Road(net, record_history=True,)
        return self

    def _create_vehicles(self):
        self.controlled_vehicles = []
        for vehicle_model in self.network_model.vehicles:
            vehicle = Vehicle(
                self.road, vehicle_model.start_position, vehicle_model.start_heading,
                vehicle_model.start_speed
            )
            vehicle.LENGTH = vehicle_model.length
            vehicle.WIDTH = vehicle_model.width
            vehicle.color = vehicle_model.color  # type: ignore

            if vehicle_model.goal:
                vehicle.goal = Landmark(  # type: ignore
                    self.road,
                    vehicle_model.goal.position.tolist(),
                    heading=vehicle_model.goal.heading,
                    speed=vehicle_model.goal.speed
                )
                self.road.objects.append(vehicle.goal)  # type: ignore

            self.road.vehicles.append(vehicle)
            if vehicle_model.is_ego:
                self.controlled_vehicles.append(vehicle)

        return self

    def _create_obstacles(self):
        for obstacle_model in self.network_model.obstacles:
            obstacle = Obstacle(
                self.road,
                obstacle_model.position.tolist(),
                obstacle_model.heading
            )
            obstacle.LENGTH = obstacle_model.length
            obstacle.WIDTH = obstacle_model.width
            obstacle.diagonal = (obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)
        return self
    # endregion

    def _convert_observation(self, observation: ObservationDict) -> Tensor:
        """将 highway-env 返回的字典格式观测转为 Tenor"""
        ob = observation["observation"]
        achieved_goal = observation["achieved_goal"]

        # assert ob == achieved_goal
        desired_goal = observation["desired_goal"]
        return concatenate([from_numpy(ob), from_numpy(desired_goal)], dim=0)
