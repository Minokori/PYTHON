import torch
from torch.optim.lr_scheduler import MultiStepLR

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import (IAgentOptimizer, IAgentScheduler,
                                        IOptimizer)


class MultiStepScheduler(MultiStepLR):
    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        super().__init__(optimizer=optimizer.optimizer, milestones=config.milestones, gamma=config.gamma)
        self._config = config

    @property
    def config(self) -> HyperParameterConfig:
        return self._config

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self  # type: ignore


class AgentMultiStepScheduler(IAgentScheduler):
    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        assert isinstance(optimizer, IAgentOptimizer), "optimizer must be an instance of IAgentOptimizer"
        self._config = config
        self._actor_scheduler = MultiStepLR(optimizer=optimizer.actor_optimizer, milestones=config.milestones, gamma=config.gamma)
        self._critic_scheduler = MultiStepLR(optimizer=optimizer.critic_optimizer, milestones=config.milestones, gamma=config.gamma)

    @property
    def actor_scheduler(self) -> MultiStepLR:
        return self._actor_scheduler

    @property
    def critic_scheduler(self) -> MultiStepLR:
        return self._critic_scheduler

    @property
    def config(self) -> HyperParameterConfig:
        return self._config

    @property
    def scheduler(self) -> MultiStepLR:
        return self._actor_scheduler
