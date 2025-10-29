from torch.optim import AdamW as AdamW
from torch.optim import Optimizer

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import IAgentOptimizer, IOptimizer
from modelsolver.abc.model import IAgentModel, IModel


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
    def optimizer(self) -> Optimizer:
        """返回优化器实例"""
        return self


class AgentAdamWOptimizer(IAgentOptimizer):
    def __init__(self, model: IModel, config: HyperParameterConfig):
        self._config = config
        assert isinstance(model, IAgentModel), "model 必须是 IAgentModel 的实例"
        self._actor_optimizer = AdamW(model.actor.parameters(),
                                      lr=config.actor_lr,
                                      betas=config.betas,
                                      eps=config.eps,
                                      weight_decay=config.weight_decay,
                                      )

        self._critic_optimizer = AdamW(model.critic.parameters(),
                                       lr=config.critic_lr,
                                       betas=config.betas,
                                       eps=config.eps,
                                       weight_decay=config.weight_decay,
                                       )

        self._critic_other_optimizer = AdamW(model.critic.parameters(),
                                             lr=config.critic_lr,
                                             betas=config.betas,
                                             eps=config.eps,
                                             weight_decay=config.weight_decay,
                                             )

        self._log_alpha_optimizer = AdamW([model.log_alpha],
                                          lr=config.learning_rate,
                                          betas=config.betas,
                                          eps=config.eps,
                                          weight_decay=config.weight_decay,
                                          )

    @property
    def config(self): return self._config

    @property
    def actor_optimizer(self) -> Optimizer:
        return self._actor_optimizer

    @property
    def critic_optimizer(self) -> Optimizer:
        return self._critic_optimizer

    @property
    def critic_other_optimizer(self) -> Optimizer:
        return self._critic_other_optimizer

    @property
    def optimizer(self) -> Optimizer:
        raise NotImplementedError("IAgentOptimizer 不实现 optimizer, 请使用 actor_optimizer 或 critic_optimizer")

    @property
    def log_alpha_optimizer(self) -> Optimizer:
        return self._log_alpha_optimizer
