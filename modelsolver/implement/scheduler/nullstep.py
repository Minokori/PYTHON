import torch
from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import IOptimizer, IScheduler


class NullScheduler(IScheduler, torch.optim.lr_scheduler.LambdaLR):
    """恒定学习率调度器"""

    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        self._config = config
        super().__init__(optimizer=optimizer.optimizer, lr_lambda=lambda epoch: self.config.learning_rate)

    @property
    def config(self) -> HyperParameterConfig:
        return self._config

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self  # type: ignore
