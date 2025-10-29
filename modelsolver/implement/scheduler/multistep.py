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
        self._critic_other_scheduler = MultiStepLR(optimizer=optimizer.critic_other_optimizer, milestones=config.milestones, gamma=config.gamma)
        self._log_alpha_scheduler = MultiStepLR(optimizer=optimizer.log_alpha_optimizer, milestones=config.milestones, gamma=config.gamma)
    @property
    def actor_scheduler(self) -> MultiStepLR:
        return self._actor_scheduler

    @property
    def critic_scheduler(self) -> MultiStepLR:
        return self._critic_scheduler

    @property
    def critic_other_scheduler(self) -> MultiStepLR:
        return self._critic_other_scheduler

    @property
    def log_alpha_scheduler(self) -> MultiStepLR:
        return self._log_alpha_scheduler

    @property
    def config(self) -> HyperParameterConfig:
        return self._config

    @property
    def scheduler(self) -> MultiStepLR:
        raise NotImplementedError("IAgentScheduler 不实现 scheduler, 请使用 actor_scheduler 或 critic_scheduler")
