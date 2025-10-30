from torch.optim import AdamW, Optimizer,Adam

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.functional import IAgentOptimizer, IOptimizer
from modelsolver.abc.model import IAgentModel, IModel


class AdamWOptimizer(IOptimizer):
    """AdamW 优化器实现类"""

    def __init__(self, model: IModel, config: HyperParameterConfig):
        self._config = config
        self._optimizer = AdamW(
                       params=model.parameters(),
                       lr=config.learning_rate,
                       betas=config.betas,
                       eps=config.eps,
                       weight_decay=config.weight_decay,
                       )
    @property
    def optimizer(self) -> Optimizer:
        """返回优化器实例"""
        return self._optimizer


class AgentAdamWOptimizer(IAgentOptimizer):
    def __init__(self, model: IModel, config: HyperParameterConfig):
        assert isinstance(model, IAgentModel), "model 必须是 IAgentModel 的实例"
        self._config = config
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

        self._critic_other_optimizer = AdamW(model.other_critic.parameters(),
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
    def actor_optimizer(self) -> Optimizer:
        return self._actor_optimizer

    @property
    def critic_optimizer(self) -> Optimizer:
        return self._critic_optimizer

    @property
    def critic_other_optimizer(self) -> Optimizer:
        return self._critic_other_optimizer
    @property
    def log_alpha_optimizer(self) -> Optimizer:
        return self._log_alpha_optimizer
