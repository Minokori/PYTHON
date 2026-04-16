"""包装后的 highway-env 停车环境"""
# region import
import logging
from typing import TYPE_CHECKING, TypedDict

import torch
from highway_env.envs import ParkingEnv
from highway_env.envs.common.observation import \
    KinematicsGoalObservation as _KinematicsGoalObservation
from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle
from numpy import bool_, float32
from numpy.typing import NDArray
from torch import Tensor, concatenate, from_numpy, no_grad, tensor

from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.reward import IReward
from warppedhighway.config import RoadNetworkModel


logging.basicConfig(level=logging.DEBUG, filename="parking.log", filemode="w", encoding="utf-8")

# endregion

# region 环境配置字典
class ActionType(TypedDict):
    type: str


class ObservationType(TypedDict):
    type: str
    features: list[str]
    scales: list[float]
    normalize: bool

class InfoType(TypedDict):
    speed:float
    crashed:bool
    action:NDArray[float32]
    is_success:bool_

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
class ParkingReward(IReward):
    WEIGHT = torch.tensor([1.0, 1.0, 0.01, 0.01, 0.5, 0.5]).reshape(-1,1)
    # WEIGHT = torch.tensor([1.0, 1.0, 0.01, 0.01, 0.05, 0.05]).reshape(-1,1)
    """各状态分量在计算奖励时的权重."""
    @no_grad
    def forward(self, **kwargs:Tensor) -> tuple[Tensor, Tensor]:
        # 解压数据
        state = kwargs["state"]
        batched = len(state.shape)> 1  # env单步计算时形状为1维度, 经验回放池批量计算时形状为2维度.
        state = state.reshape(-1,12)
        action = kwargs["action"].reshape(-1,2)
        next_state = kwargs["next_state"].reshape(-1,12)

        # 计算奖励 s, g: x, y, vx, vy, cos_h, sin_h | a: steering, acceleration

        # 已到达的state和目标state之间的距离, 以及速度和朝向的差异
        diff  = (next_state[:,0:6] - next_state[:,6:12]).abs() # (B,6)
        theta_diff = torch.rad2deg(
            (
                torch.atan2(next_state[:,5], next_state[:,4]) -
                torch.atan2(next_state[:,11], next_state[:,10])
                ).abs()
                ).reshape(-1,)
        # reward = - (diff.pow(2) @ self.WEIGHT).pow(0.5)  # (B,1)
        reward = - (diff @ self.WEIGHT).pow(0.5)  # (B,1)

        # 是否到达目标位置的判断
        # TODO 应该更新
        # done = torch.where(
        #     reward > -0.12,
        #     torch.full_like(reward, 1.0),
        #     torch.full_like(reward, 0.0)).float() # (B,1)
        done = self._is_done(diff,theta_diff)

        # reward += done * 200  # 到达目标位置的奖励
        reward += done * 5  # 到达目标位置的奖励

        if not batched:
            reward = reward.reshape(1)
            done = done.reshape(1)
        else:
            reward = reward.reshape(-1,1)
            done = done.reshape(-1,1)
        return reward, done

    def _is_done(self, diff:Tensor, theta_diff:Tensor)-> Tensor:
        """判断是否到达目标位置"""
        # ±5° → ≈ 0.087
        # ±8° → ≈ 0.140
        # ±10° → ≈ 0.174
        # ±12° → ≈ 0.209
        # ±15° → ≈ 0.261
        # ±20° → ≈ 0.347
        # diff.shape = (B,6)
        x_diff = diff[:,0]
        y_diff = diff[:,1]
        vx_diff = diff[:,2]
        vy_diff = diff[:,3]

        position_diff  = (x_diff**2 + y_diff**2).pow(0.5)
        speed_diff     = (vx_diff**2 + vy_diff**2).pow(0.5)


        total_diff = (position_diff<0.2) * (speed_diff<0.1) * (theta_diff<10)

        return total_diff.float().reshape(-1,1)

    @property
    def is_learnable(self) -> bool:
        return False

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
        "collision_reward": -50,
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
    # endregion

    def __init__(self, config: RoadNetworkModel = RoadNetworkModel(),reward: IReward = ParkingReward()):
        self._config = config
        self._reward_function = reward
        ParkingEnv.__init__(self,dict(self.DEFAULT_CONFIG), render_mode="human")


    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        state = self._convert_observation_to_tensor(self.observation_type_parking.observe())
        next_observation, reward, terminated, truncated, info = super(ParkingEnv, self).step(action.cpu().detach().numpy())
        next_state = self._convert_observation_to_tensor(next_observation)


        reward, done = self._reward_function(state=state, action=action, next_state=next_state)
        if done:
            logging.info(f"成功!. 状态:{next_state.tolist()}, 奖励:{reward.item()}")

        if info["crashed"]:
            reward -= 200
            done = torch.tensor(1.0).float()

        return next_state,reward,done,truncated,info


    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        observation, info = super(ParkingEnv, self).reset()
        state = self._convert_observation_to_tensor(observation)
        reward ,done= self._reward_function(state=state, action=self.ZERO_ACTION, next_state=state)
        return state, reward, done, False, info
    # endregion



    # region override private methods (创建路网)
    def _reset(self):
        if not self._config.lanes:
            super()._reset()
            return
        self._create_road()\
            ._create_vehicles()\
            ._create_obstacles()

    def _create_road(self):
        """构建路网"""
        if not self._config.lanes:
            super()._create_road()
            return self
        net = RoadNetwork()

        # add lanes
        for lane_model in self._config.lanes:
            net.add_lane(
                lane_model.from_node,
                lane_model.to_node,
                StraightLane(
                    lane_model.start_position,
                    lane_model.end_position,
                    lane_model.width,
                    line_types=lane_model.line_types, # type: ignore
                ),
            )

        # create road network
        self.road = Road(net, record_history=True,)
        return self

    def _create_vehicles(self):
        if not self._config.vehicles:
            super()._create_vehicles()
            return self
        self.controlled_vehicles = []
        for vehicle_model in self._config.vehicles:
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
                    vehicle_model.goal.position,
                    heading=vehicle_model.goal.heading,
                    speed=vehicle_model.goal.speed
                )
                self.road.objects.append(vehicle.goal)  # type: ignore

            self.road.vehicles.append(vehicle)
            if vehicle_model.is_ego:
                self.controlled_vehicles.append(vehicle)

        return self

    def _create_obstacles(self):
        if not self._config.obstacles:
            return self
        for obstacle_model in self._config.obstacles:
            obstacle = Obstacle(
                self.road,
                obstacle_model.position,
                obstacle_model.heading
            )
            obstacle.LENGTH = obstacle_model.length
            obstacle.WIDTH = obstacle_model.width
            obstacle.diagonal = (obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)
        return self
    # endregion


    # 其他私有方法
    def _convert_observation_to_tensor(self, observation_dict: ObservationDict) -> Tensor:
        """将 highway-env 返回的字典格式观测转为 Tensor (S, GOAL)"""
        observation = observation_dict["observation"]
        achieved_goal = observation_dict["achieved_goal"]

        # assert observation == achieved_goal
        desired_goal = observation_dict["desired_goal"]
        return concatenate([from_numpy(observation.copy()), from_numpy(desired_goal.copy())], dim=0).float()


    # 属性
    @property
    def ZERO_ACTION(self) -> Tensor:
        return tensor([0.0, 0.0]).float()
    @property
    def GOAL(self) -> Tensor:
        return tensor(self.road.objects[0].position)  # type: ignore










__all__ = ["ParkingEnvironment"]