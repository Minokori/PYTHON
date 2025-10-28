from torch.optim import AdamW, Optimizer

from modelsolver.abc.config import HyperParameterConfig
from modelsolver.abc.model import IModel
from modelsolver.abc.rl.model import IAgentModel
from modelsolver.abc.rl.optimizier import IAgentOptimizer


class DefaultAgentOptimizer(IAgentOptimizer):
    def __init__(self, model: IModel, config: HyperParameterConfig):
        self._config = config
        assert isinstance(model, IAgentModel), "model 必须是 IAgentModel 的实例"
        self._actor_optimizer = AdamW(model.actor.parameters(), lr=config.learning_rate,
                                      betas=config.betas,
                                      eps=config.eps,
                                      weight_decay=config.weight_decay,
                                      )

        self._critic_optimizer = AdamW(model.critic.parameters(), lr=config.learning_rate, betas=config.betas,
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
    def optimizer(self) -> Optimizer:
        return self.actor_optimizer
