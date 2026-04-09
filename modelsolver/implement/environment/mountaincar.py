# regiopn imports
import logging
from typing import Self

import numpy as np
from gymnasium.envs.classic_control import Continuous_MountainCarEnv
from torch import Tensor, from_numpy, tensor

from modelsolver.abc.environment import IEnvironment


logging.basicConfig(level=logging.DEBUG, filename="mountaincar.log", filemode="w",encoding="utf-8")
np.set_printoptions(precision=2, suppress=True)
# endregion




class MountainCarEnvironment(Continuous_MountainCarEnv,IEnvironment):
    """山地车环境"""

    def __init__(self) -> None:
        super().__init__(render_mode="human")
        open("mountaincar.log", "w").close()  # 清空日志文件

    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, info = super().reset()
        state = from_numpy(ob.copy()).float().reshape(-1)
        # 归一化 state
        state[0] = (state[0] + 0.3) / 0.9  # 位置归一化到 [-1, 1]
        state[1] = (state[1] / 0.07) # 速度归一化到 [-1, 1]
        return state, tensor(0).float().reshape(1), tensor(0).float().reshape(1), False, info

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict]:
        ob, reward, terminated, truncated, info = super().step(action.cpu().detach().numpy())
        state = from_numpy(ob.copy()).float().reshape(-1)
        # 归一化 state
        state[0] = (state[0] + 0.3) / 0.9  # 位置归一化到 [-1, 1]
        state[1] = (state[1] / 0.07) # 速度归一化到 [-1, 1]



        return state, tensor(reward*10).float().reshape(1), tensor(terminated).float().reshape(1), truncated, info

    def build_environment(self) -> Self:
        return self
