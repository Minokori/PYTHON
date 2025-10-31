from torch.optim import AdamW, Optimizer

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

    def __getitem__(self, key: str) -> Optimizer:
        match key:
            case "all" | "":
                return self._optimizer
            case _:
                raise KeyError(f"Unknown optimizer key: {key}")


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

    def __getitem__(self, key: str) -> Optimizer:
        match key:
            case "actor":
                return self._actor_optimizer
            case "critic":
                return self._critic_optimizer
            case "critic_other":
                return self._critic_other_optimizer
            case "log_alpha":
                return self._log_alpha_optimizer
            case _:
                raise KeyError(f"Unknown optimizer key: {key}")
