from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import (IAgentOptimizer, IAgentScheduler,
                                        IOptimizer, IScheduler)


class NullScheduler(IScheduler):
    """恒定学习率调度器"""

    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        self._config = config
        self._scheduler = LambdaLR(optimizer=optimizer["all"], lr_lambda=lambda epoch: self.config.learning_rate)

    @property
    def config(self) -> HyperParameterConfig:
        return self._config

    def __getitem__(self, key: str) -> LambdaLR:
        match key:
            case "all" | "":
                return self._scheduler  # type: ignore
            case _:
                raise KeyError(f"Unknown Scheduler key: {key}")



class AgentNullScheduler(IAgentScheduler):
    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        assert isinstance(optimizer, IAgentOptimizer), "optimizer must be an instance of IAgentOptimizer"
        self._config = config
        self._actor_scheduler = LambdaLR(optimizer=optimizer.actor_optimizer, lr_lambda=lambda epoch: self.config.actor_lr)
        self._critic_scheduler = LambdaLR(optimizer=optimizer.critic_optimizer, lr_lambda=lambda epoch: self.config.critic_lr)
        self._critic_other_scheduler = LambdaLR(optimizer=optimizer.critic_other_optimizer, lr_lambda=lambda epoch: self.config.critic_lr)
        self._log_alpha_scheduler = LambdaLR(optimizer=optimizer.log_alpha_optimizer, lr_lambda=lambda epoch: self.config.learning_rate)


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
    def actor_scheduler(self) -> LambdaLR:
        return self._actor_scheduler

    @property
    def critic_scheduler(self) -> LambdaLR:
        return self._critic_scheduler

    @property
    def critic_other_scheduler(self) -> LambdaLR:
        return self._critic_other_scheduler

    @property
    def log_alpha_scheduler(self) -> LambdaLR:
        return self._log_alpha_scheduler

    @property
    def config(self) -> HyperParameterConfig:
        return self._config
