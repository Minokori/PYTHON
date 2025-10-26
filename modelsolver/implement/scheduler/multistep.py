import torch

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import IOptimizer, IScheduler


class MultiStepScheduler(IScheduler, torch.optim.lr_scheduler.MultiStepLR):
    def __init__(self, config: HyperParameterConfig, optimizer: IOptimizer):
        super().__init__(optimizer=optimizer.optimizer, milestones=config.milestones, gamma=config.gamma)
        self._config = config

    @property
    def config(self) -> HyperParameterConfig:
        return self._config

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self  # type: ignore