from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import (IAgentOptimizer, IAgentScheduler,
                                        IOptimizer, IScheduler)


class NullScheduler(IScheduler, LambdaLR):
    """恒定学习率调度器"""

    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        self._config = config
        super().__init__(optimizer=optimizer.optimizer, lr_lambda=lambda epoch: self.config.learning_rate)

    @property
    def config(self) -> HyperParameterConfig:
        return self._config

    @property
    def scheduler(self) -> _LRScheduler:
        return self  # type: ignore


class AgentNullScheduler(IAgentScheduler):
    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        assert isinstance(optimizer, IAgentOptimizer), "optimizer must be an instance of IAgentOptimizer"
        self._config = config
        self._actor_scheduler = LambdaLR(optimizer=optimizer.actor_optimizer, lr_lambda=lambda epoch: self.config.actor_lr)
        self._critic_scheduler = LambdaLR(optimizer=optimizer.critic_optimizer, lr_lambda=lambda epoch: self.config.critic_lr)
        self._critic_other_scheduler = LambdaLR(optimizer=optimizer.critic_other_optimizer, lr_lambda=lambda epoch: self.config.critic_lr)
        self._log_alpha_scheduler = LambdaLR(optimizer=optimizer.log_alpha_optimizer, lr_lambda=lambda epoch: self.config.learning_rate)

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
