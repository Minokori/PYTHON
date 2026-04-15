# regiopn imports
import logging

import numpy as np
from gymnasium.envs.box2d import LunarLander
from torch import Tensor, concat, from_numpy, tensor

from modelsolver.abc.config import EnvironmentConfig
from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.reward import IReward


logging.basicConfig(level=logging.DEBUG, filename="lunarlander.log", filemode="w",encoding="utf-8")
np.set_printoptions(precision=2, suppress=True)
# endregion


class LunarLanderEnvironment(LunarLander, IEnvironment):
    """漫步环境"""

    def __init__(self, config:EnvironmentConfig, reward: IReward) -> None:
        self._config = config
        self._reward_function = reward
        self._env=LunarLander(render_mode="human", continuous=True, enable_wind=True)
        open("lunarlander.log", "w").close()  # 清空日志文件


    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, info = self._env.reset()
        return concat([from_numpy(ob).float().reshape(-1),self.GOAL]), tensor(0).float().reshape(1), tensor(0).float().reshape(1), False, info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, reward, terminated, truncated, info = self._env.step(action.cpu().detach().numpy())
        state = from_numpy(ob).float().reshape(-1)
        return concat([state, self.GOAL]), tensor(reward).float().reshape(1), tensor(terminated).float().reshape(1), truncated, info

    @property
    def ZERO_ACTION(self) -> Tensor:
        return tensor([0.0, 0.0]).float()

    @property
    def GOAL(self) -> Tensor:
        # x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact
        goal_x = (self._env.helipad_x1-10 + self._env.helipad_x2-10)/10
        goal_y = (self._env.helipad_y- (self._env.helipad_y+18/30))/(400/60)
        goal_vx = 0.0
        goal_vy = 0.0
        goal_angle = 0.0
        goal_angular_velocity = 0.0
        goal_left_leg_contact = 1.0
        goal_right_leg_contact = 1.0
        return tensor([goal_x, goal_y, goal_vx, goal_vy, goal_angle, goal_angular_velocity, goal_left_leg_contact, goal_right_leg_contact]).float()

