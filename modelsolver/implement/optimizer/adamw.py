import torch
from torch.optim import AdamW as AdamW

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import IOptimizer
from modelsolver.abc.model import IModel


class AdamWOptimizer(IOptimizer, AdamW):
    """AdamW 优化器实现类"""

    def __init__(self, model: IModel, config: HyperParameterConfig):
        self._config = config
        AdamW.__init__(self,
                       params=model.parameters(),
                       lr=config.learning_rate,
                       betas=config.betas,
                       eps=config.eps,
                       weight_decay=config.weight_decay,
                       )

    @property
    def config(self) -> HyperParameterConfig:
        """优化器配置"""
        return self._config

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """返回优化器实例"""
        return self
