from abc import ABC, abstractmethod

from torch.optim import Optimizer

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import IOptimizer
from modelsolver.abc.model import IModel
from modelsolver.abc.rl.model import IAgentModel


class IAgentOptimizer(IOptimizer, ABC):
    """智能体优化器接口.

    必须为 Actor 和 Critic 分别提供优化器
    """

    def __init__(self, model: IModel, config: HyperParameterConfig):
        assert isinstance(model, IAgentModel), "model 必须是 IAgentModel 的实例"
        self._config = config

    @property
    @abstractmethod
    def config(self) -> HyperParameterConfig:
        return self._config

    @property
    @abstractmethod
    def actor_optimizer(self) -> Optimizer:
        """返回 actor 的优化器"""
        raise NotImplementedError

    @property
    @abstractmethod
    def critic_optimizer(self) -> Optimizer:
        """返回 critic 的优化器"""
        raise NotImplementedError
