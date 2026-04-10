# regiopn imports
import logging
from typing import Self

import numpy as np
from gymnasium.envs.box2d import LunarLander
from torch import Tensor, from_numpy, tensor

from modelsolver.abc.environment import IEnvironment


logging.basicConfig(level=logging.DEBUG, filename="lunarlander.log", filemode="w",encoding="utf-8")
np.set_printoptions(precision=2, suppress=True)
# endregion


class LunarLanderEnvironment(IEnvironment):
    """漫步环境"""

    def __init__(self) -> None:
        self._env = LunarLander(render_mode="human",continuous=True,enable_wind=True)
        open("lunarlander.log", "w").close()  # 清空日志文件


    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, info = self._env.reset()
        return from_numpy(ob).float().reshape(-1), tensor(0).float().reshape(1), tensor(0).float().reshape(1), False, info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, reward, terminated, truncated, info = self._env.step(action.cpu().detach().numpy())
        state = from_numpy(ob).float().reshape(-1)
        return state, tensor(reward).float().reshape(1), tensor(terminated).float().reshape(1), truncated, info

    def build_environment(self) -> Self:
        return self
