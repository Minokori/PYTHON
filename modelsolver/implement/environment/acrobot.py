from typing import Self

import torch
from gymnasium.envs.classic_control import AcrobotEnv
from torch import Tensor, floor, tensor

from modelsolver.abc.environment import IEnvironment


class AcrobotEnvironment( AcrobotEnv,IEnvironment):
    """对 Acrobot 环境的包装, 接受**连续**动作

    **动作映射**:

    action = int(floor(action*3).item())  # 将连续动作映射到离散动作空间 [0, 1, 2]
    """
    def __init__(self) -> None:
        super().__init__(render_mode="human")


    def reset(self) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:
        ob, info = super().reset()

        return (tensor(ob.copy()).float().reshape(-1),
                 tensor(0).float().reshape(1),
                   tensor(0).float().reshape(1),
                     False,
                       info)


    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:

        a = int(floor(action*3).item())

        ob, reward, terminated, truncated, info = super().step(a)

        # 归一到 [-1, 1]
        state = tensor(ob.copy()).float().reshape(-1)
        state[4]/=torch.pi
        state[5]/=torch.pi

        return (state,
                 tensor(reward).float().reshape(1),
                 tensor(terminated).float().reshape(1),
                 truncated,
                 info)
