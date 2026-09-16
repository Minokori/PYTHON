"""对原生Highway环境的封装"""
from highway_env.envs.highway_env import HighwayEnv
from torch import Tensor

from modelsolver.abc.config import EnvironmentConfig
from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.reward import IReward


class Highway(IEnvironment):

    def __init__(self, config: EnvironmentConfig, reward: IReward) -> None:
        self._env = HighwayEnv(render_mode="human")


    def reset(self) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        raise NotImplementedError("请使用 reset_with_goal 方法重置环境, 以便在重置时指定目标状态")

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:


        # step
        next_obs, reward, terminated, truncated, info = self._env.step(action)

        raise NotImplementedError("请使用 step_with_goal 方法执行环境步进, 以便在步进时指定目标状态")