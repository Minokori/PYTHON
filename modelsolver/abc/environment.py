"""环境模块, 定义了环境接口, 包括强化学习环境接口"""
from abc import ABC, abstractmethod

from torch import Tensor

from modelsolver.abc.config import EnvironmentConfig
from modelsolver.abc.reward import IReward


# TODO step 返回的state包含goal, 方便HER算法的实现
# TODO 抽象出一个IReward接口.


class IEnvironment(ABC):


    """包装环境的接口.

    + 需要重写环境的 `reset` 和 `step` 方法, 使其返回相同的数据结构, 并且数据类型为 `Tensor.cpu()`
    + 需要重写环境的 `step` 方法, 使其接受类型为 `Tensor` 的动作

    ---

    依赖:
    + `EnvironmentConfig` : 环境配置类, 包含环境的各种参数
    + `IReward` : 奖励函数接口, 用于计算环境的奖励
    """


    @abstractmethod
    def __init__(self, config: EnvironmentConfig, reward:IReward) -> None:
        """初始化环境

        Args:
            config (EnvironmentConfig): 环境配置项
            reward (IReward): 奖励函数项
        """
        ...

    # TODO 修改 done的定义: -1表示失败, 0表示未完成, 1表示成功. 这样可以更好地支持HER等算法.
    @abstractmethod
    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:
        """执行一步环境交互, 返回 (observation, reward, terminated, truncated, info).cpu().

        + observation: 环境的观测状态, 包含 **当前状态** 和 **目标状态**, 用于HER等算法的实现. 预期得到 Tensor.cpu(), shape = (obs_dim).

            *举例: 如果一个环境返回智能体的位置(x,y)作为观测, 那么 observation 应该为 (x,y, x_goal, y_goal)*
        + reward: 环境的奖励, 预期得到 Tensor.cpu(), shape = (1,).
        + terminated(done): 环境是否终止, 预期得到 Tensor.cpu(), shape = (1,).

            * 为兼容 HER 算法, reward 和 done 都由 `IReward` 计算, 其中, done 不再为布尔值, 而是 `-1, 0, 1` 的整数值. 分别表示失败, 未完成目标, 和成功.
        + truncated(timeout): 环境是否被截断, 预期得到 Tensor.cpu(), shape = (1,).
        + info: 环境的额外信息, 预期得到 dict[str, Tensor]

        ---

        以上返回的Tensor数据类型均为 float32.

        Args:
            action (Tensor):动作. Tensor.cpu() shape = (action_dim,).

        Returns:
            环境交互信息 (tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]): 见上.
        """
        ...

    @abstractmethod
    def reset(self) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """和 `step` 保持一致, 返回 (observation, reward, terminated, truncated, info).cpu().

        *对于 reset() 而言, 仅 observation 有意义*
        """
        ...

    @property
    def ZERO_ACTION(self) -> Tensor:
        """环境的零动作, 用于经验回放池预热等场景. 预期得到 Tensor.cpu()"""
        ...

    @property
    def GOAL(self) -> Tensor:
        """环境的目标状态, 用于HER等算法. 预期得到 Tensor.cpu()"""
        ...










