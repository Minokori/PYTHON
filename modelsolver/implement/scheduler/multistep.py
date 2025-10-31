from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import (IAgentOptimizer, IAgentScheduler,
                                        IOptimizer, IScheduler)


class MultiStepScheduler(IScheduler):
    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        self._config = config
        self._scheduler = MultiStepLR(optimizer=optimizer["all"], milestones=config.milestones, gamma=config.gamma)

    @property
    def config(self) -> HyperParameterConfig:
        return self._config

    def __getitem__(self, key: str) -> LRScheduler:
        match key:
            case "all" | "":
                return self._scheduler  # type: ignore
            case _:
                raise KeyError(f"Unknown Scheduler key: {key}")

class AgentMultiStepScheduler(IAgentScheduler):
    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        assert isinstance(optimizer, IAgentOptimizer), "optimizer must be an instance of IAgentOptimizer"
        self._config = config
        self._actor_scheduler = MultiStepLR(optimizer=optimizer.actor_optimizer, milestones=config.milestones, gamma=config.gamma)
        self._critic_scheduler = MultiStepLR(optimizer=optimizer.critic_optimizer, milestones=config.milestones, gamma=config.gamma)
        self._critic_other_scheduler = MultiStepLR(optimizer=optimizer.critic_other_optimizer, milestones=config.milestones, gamma=config.gamma)
        self._log_alpha_scheduler = MultiStepLR(optimizer=optimizer.log_alpha_optimizer, milestones=config.milestones, gamma=config.gamma)

    def __getitem__(self, key: str) -> LRScheduler:
        match key:
            case "actor":
                return self._actor_scheduler
            case "critic":
                return self._critic_scheduler
            case "critic_other":
                return self._critic_other_scheduler
            case "log_alpha":
                return self._log_alpha_scheduler
            case _:
                raise KeyError(f"Unknown Scheduler key: {key}")


    @property
    def config(self) -> HyperParameterConfig:
        return self._config
