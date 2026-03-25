"""采用依赖注入 (DI) 的模型训练一键式解决方案"""

# region 库导入
import os
from dataclasses import is_dataclass
from math import floor, log
from typing import Any, Literal, Self

import torch
from clean_ioc import Container, DependencySettings, Lifespan
from clean_ioc.registration_filters import with_name
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy import mean
from torch import load, no_grad, save
from torch.autograd import set_detect_anomaly
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, random_split

from modelsolver.abc.config import DataConfig, HyperParameterConfig
from modelsolver.abc.data import (IDataLoader, IDataProcesser, IDataset,
                                  IReplayBuffer)
from modelsolver.abc.environment import IEnvironment
from modelsolver.abc.functional import (IAgentLoss, IAgentOptimizer,
                                        IAgentScheduler, ILoss, IOptimizer,
                                        IScheduler)
from modelsolver.abc.model import IActor, IAgentModel, ICritic, IModel
from modelsolver.implement.loss import DefaultAgentLoss
from modelsolver.implement.model import DefaultAgent, NullActor, NullCritic
from modelsolver.implement.optimizer.adamw import (AdamWOptimizer,
                                                   AgentAdamWOptimizer)
from modelsolver.implement.scheduler.nullstep import (AgentNullScheduler,
                                                      NullScheduler)


# endregion

# TODO model的 __call__ 方法的 **kwargs


class ModelSolver(Container):
    """模型解决方案"""

    def __init__(self):
        super().__init__()
        self.register(FontProperties, instance=FontProperties("./data/SourceHanSansLite.ttf", size=14))

        # 训练过程中的损失记录
        self.stats = self._init_stats()

        # 子 ioc 容器, 构建模型和数据加载器
        self._model_builder = Container()
        self._dataloader_builder = Container()

        # dataloader 缓存
        self._dataloaders: list[IDataLoader] = []

        # 一些组件的默认实现
        self.add_optimizer(AdamWOptimizer)
        self.add_lr_scheduler(NullScheduler)

    # region 注册损失函数, 优化器, 学习率调度器, 配置项, 数据处理器, 使 ioc 更易用
    def add_loss_function(self, loss_function: type[ILoss]):
        """注册损失函数"""
        self.register(ILoss, loss_function, lifespan=Lifespan.singleton)
        return self

    def add_optimizer(self, optimizer: type[IOptimizer]):
        """注册优化器.

        + 若没有指定优化器, 则使用 AdamW 作为默认优化器.
        """
        self.register(IOptimizer, implementation_type=optimizer, lifespan=Lifespan.singleton)
        return self

    def add_lr_scheduler(self, lr_scheduler: type[IScheduler]):
        """注册学习率调度器

        + 若没有指定学习率调度器, 则使用 NullScheduler 作为默认学习率调度器. (不会动态调整学习率)
        """
        self.register(IScheduler, lr_scheduler, lifespan=Lifespan.singleton)
        return self

    def add_config(self, config: Any, name: str | None = None):
        """注册配置."""
        assert is_dataclass(config), "config 不是 dataclass"
        self.register(type(config), instance=config, lifespan=Lifespan.singleton, name=name)
        return self

    def add_data_processer(self, processer: type[IDataProcesser]):
        """注册数据处理器"""
        self._dataloader_builder.register(IDataProcesser, implementation_type=processer, lifespan=Lifespan.singleton)
        return self
    # endregion

    # region 注册模型
    # 模型有两种注册方式
    # 1. 直接注册模型实例
    # 2. 注册模型组件, 由 ioc 容器构建模型实例
    def add_model(self, model: IModel | type[IModel]):
        """注册模型实例或类型"""
        match model:
            case type():
                self._model_builder.register(IModel, implementation_type=model, lifespan=Lifespan.singleton)
            case IModel():
                self.register(IModel, instance=model, lifespan=Lifespan.singleton)
        return self

    def add_model_component(self, module: type, implementation: type, name: str | None = None, singleton: bool = False):
        """注册模型组件, 通过 ioc 容器构建模型实例"""
        if name:
            self._model_builder.register(module,
                                         implementation,
                                         lifespan=Lifespan.singleton if singleton else Lifespan.transient, dependency_config={
                                             "config": DependencySettings(
                                                 filter=with_name(name))})
        else:
            self._model_builder.register(module,
                                         implementation,
                                         lifespan=Lifespan.singleton if singleton else Lifespan.transient)
        return self

    def add_model_config(self, config: Any, name: str | None = None):

        assert is_dataclass(config), "config 不是 dataclass"
        self._model_builder.register(type(config), instance=config, lifespan=Lifespan.singleton, name=name)
        return self

    # endregion

    # region 注册数据来源
    def add_dataset(self, dataset: type[IDataset]):
        self._dataloader_builder.register(IDataset, dataset, lifespan=Lifespan.singleton)
        return self

    def add_data_config(self, config: DataConfig):
        if issubclass(type(config), DataConfig) and type(config) != DataConfig: # 自己实现的DataConfig类
            self._dataloader_builder.register(type(config), instance=config, lifespan=Lifespan.singleton)
            self._dataloader_builder.register(DataConfig, instance=config, lifespan=Lifespan.singleton)
            # 目前好像只有这一种方法, 经过检验所注册单例的id(), 确认不会创建副本.
        else:
            self._dataloader_builder.register(DataConfig, instance=config, lifespan=Lifespan.singleton)
        return self

    def _build_dataloader(self):
        config = self._dataloader_builder.resolve(DataConfig)

        ratio = config.ratio
        ratio_num = [floor(r * len(self.dataset)) for r in ratio]  # type: ignore
        ratio_num[-1] = len(self.dataset) - sum(ratio_num[:-1])  # type: ignore
        batch_size = config.batch_size
        self._dataloaders = [
            IDataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.data_processer.collate_fn)
            for dataset in random_split(self.dataset, ratio_num)
        ]
    # endregion

    # region properties, 使 ioc 更易用

    @property
    def config(self) -> HyperParameterConfig:
        return self.resolve(HyperParameterConfig)  # type: ignore

    @property
    def useGPU(self):
        return True

    @property
    def model(self):
        match self.has_registration(IModel), self.useGPU:
            case True, True:
                return self.resolve(IModel).cuda()
            case True, False:
                return self.resolve(IModel)

            # 组装模型
            case False, True:
                model = self._model_builder.resolve(IModel)
                self.add_model(model)
                return model.cuda()
            case False, False:
                model = self._model_builder.resolve(IModel)
                self.add_model(model)
                return model

    @property
    def optimizer(self) -> Optimizer:
        return self.resolve(IOptimizer)["all"]

    @property
    def scheduler(self):
        return self.resolve(IScheduler)["all"]

    @property
    def loss_function(self):
        return self.resolve(ILoss)

    @property
    def train_dataloader(self) -> IDataLoader:
        if not self._dataloaders:
            self._build_dataloader()
        return self._dataloaders[0]

    @property
    def test_dataloader(self):
        if not self._dataloaders:
            self._build_dataloader()
        return self._dataloaders[1]

    @property
    def valid_dataloader(self):
        if not self._dataloaders:
            self._build_dataloader()
        return self._dataloaders[2]

    @property
    def font(self):
        return self.resolve(FontProperties)

    @property
    def data_processer(self):
        return self._dataloader_builder.resolve(IDataProcesser)

    @property
    def dataset(self) -> Dataset:
        return self._dataloader_builder.resolve(IDataset)  # type: ignore

    @property
    def train_losses(self) -> list[float]:
        return self.stats["train_loss"]

    @property
    def test_losses(self) -> list[float]:
        return self.stats["test_loss"]
    # endregion

    def _init_stats(self) -> dict[str, list[float]]:
        return {
            "train_loss": [],
            "test_loss": [],
        }

    # region 训练方法
    def train_single_epoch(self, epoch: int):
        # set to training mode
        self.model.train()

        # init stats
        total_loss = .0
        loss_num = 0

        # iterate over train dataloader
        for train_batch in self.train_dataloader:

            # preprocess data if need
            sample_batch, label_batch = self.data_processer.preprocess(train_batch)

            # load data to device
            if self.useGPU:
                sample_batch = [i.cuda().float() for i in sample_batch]
                label_batch = [i.cuda().float() for i in label_batch]

            # zero grad
            self.optimizer.zero_grad()

            # forward
            # TODO 如何让 model 的 关键字参数和自定义 IModel接受的其他参数匹配
            predict_state, label_state, state_length = self.model(sample_batch, state=label_batch)

            # compute loss
            loss = self.loss_function(predict_state, label_state, length=state_length)

            # backward
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            loss.backward()

            # update parameters
            self.optimizer.step()

            # stats
            total_loss += loss.item() * len(label_batch)  # 累计损失
            loss_num += len(label_batch)  # 累计样本数

        # update lr
        if self.scheduler:
            self.scheduler.step()

        return total_loss / loss_num

    def train(self, print_interval: int = 0, **kwargs):
        # 初始化记录参数
        str_length = round(log(self.config.epoch, 10)) + 1
        for l in self.stats.values():
            l.clear()

        with set_detect_anomaly(True, True):
            for epoch in range(self.config.epoch):
                # TODO 现在是训练损失->step->test损失, 会导致最后一个epoch的训练损失和测试损失不在同一个step上, 需要调整为训练损失->测试损失->step
                test_loss, other_stats = self._evaluate()
                train_loss = self.train_single_epoch(epoch)
                

                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)

                if print_interval and (epoch + 1) % print_interval == 0:
                    print(
                        f"Epoch: {str(epoch + 1).zfill(str_length)}, Train loss: {
                            train_loss:.4f}, Test loss: {
                            test_loss:.4f}, lr: {
                            self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}"
                    )

    # endregion

    @no_grad()
    def _evaluate(self):
        # set to evaluation mode
        self.model.eval()

        # init stats
        total_loss = .0
        loss_num = 0

        # iterate over test dataloader
        for test_batch in self.test_dataloader:
            # preprocess data if need
            sample_batch, state_batch = self.data_processer.preprocess(test_batch)

            # load data to device
            if self.useGPU:
                sample_batch = [seq.cuda().float() for seq in sample_batch]
                state_batch = [state.cuda().float() for state in state_batch]

            predict_state, label_state, length = self.model(sample_batch, state=state_batch)

            loss = self.loss_function(predict_state, label_state, length=length)

            total_loss += loss.item() * len(state_batch)
            loss_num += len(state_batch)
        return total_loss / loss_num, {}

    def save_model(self, dir: str):
        """保存模型参数到指定文件夹

        Args:
            dir (str): 文件夹名(不需要以`'/'`结尾)
        """
        paras = self.model.state_dict()
        r = []
        for k, v in paras.items():
            if "reverse" in k:
                r.append(k)
        for k in r:
            del paras[k]

        save(paras, f"{dir}/{self.model.name_for_save}.pth")

    def load_model(self, path: str):
        """从指定路径加载模型参数

        Args:
            path (str): 模型参数文件路径
        """
        if self.useGPU:
            self.model.load_state_dict(load(path, map_location='cuda'))
        else:
            self.model.load_state_dict(load(path))

    # region 扩展方法
    @no_grad()
    def evalute_plot(self, save_suffix: str = "", save_dir: str = ""):
        """评估模型并绘制预测结果
        该方法会遍历训练集、测试集和验证集，绘制每个样本的预测结果与真实标签的对比图。

        Args:
            save_suffix (str, optional): 图片保存后缀. Defaults to "".

            若为空, 则不会保存, 直接显示
        """
        for name, dataloader in zip(["train", "test", "valid"], [self.train_dataloader,
                                                                 self.test_dataloader, self.valid_dataloader]):
            count = 0
            for batch in dataloader:
                inputs, labels = self.data_processer.preprocess(batch)
                inputs = [i.cuda().float() for i in inputs]
                labels = [i.cuda().float() for i in labels]

                label_hats, _, lengths = self.model(inputs, state=labels)
                label_hats = self.model.reverse(label_hats)  # type: ignore

                for label_hat, label, length in zip(label_hats, labels, lengths):

                    y_hat = self.data_processer.postprocess(label_hat, target="label")
                    y = self.data_processer.postprocess(label, target="label")

                    keys = y_hat.keys()
                    fig, axes = plt.subplots(len(keys), 1)  # type: ignore

                    x = [i + 1 for i in range(length)]

                    if len(keys) == 1:
                        axes: list[plt.Axes] = [axes]  # type: ignore
                        keys = [list(keys)[0]]

                    for ax, key in zip(axes, keys):
                        ax.plot(x, y_hat[key].squeeze(), label=f"预测 {key}"
                                )
                        ax.plot(x, y[key].squeeze(), label=f"真实 {key}"
                                )
                        ax.legend(prop=self.font)

                    file_name = f"{count}_{self.model.name_for_save}_{save_suffix}.png" if save_suffix else f"{count}_{self.model.name_for_save}.png"
                    save_path = f"./output/{save_dir}/{name}" if save_dir else f"./output/{name}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    if save_suffix:
                        plt.savefig(f"{save_path}/{file_name}", format="png")
                        plt.close()
                    else:
                        plt.show()
                    count += 1
                    count += 1

    def plot_losses(self,):
        x = [i + 1 for i in range(self.config.epoch)]  # type: ignore
        plt.plot(x, self.train_losses, label="训练集平均loss")
        plt.plot(x, self.test_losses, label="测试集平均loss")
        plt.legend(prop=self.font)  # type: ignore
        plt.show()
    # endregion


class AgentModelSolver(ModelSolver):

    # region environment 相关函数, override 函数签名 (提供类型检查)
    def __init__(self):
        super().__init__()
        # 初始化训练和测试奖励列表
        self.stats["rewards"] = []
        self.stats["actor_losses"] = []
        self.stats["critic_losses"] = []
        self._environment_builder = Container()

        # 默认 Actor 和 Critic 的占位符
        self.add_model_component(IActor, NullActor)
        self.add_model_component(ICritic, NullCritic)
        self.add_model_component(IModel, DefaultAgent, singleton=True)

        # 默认的 optimizer
        self.add_optimizer(AgentAdamWOptimizer)

        # 默认的 replay buffer
        self.add_replay_buffer(IReplayBuffer)

        # 默认的 loss function
        self.add_loss_function(DefaultAgentLoss)

        # 默认的 scheduler
        self.add_lr_scheduler(AgentNullScheduler)

    def add_actor_component(self, actor: type[IActor]):
        self.add_model_component(IActor, actor)
        return self

    def add_critic_component(self, critic: type[ICritic]):
        self.add_model_component(ICritic, critic)
        return self

    def add_model(self, model: IAgentModel | type[IAgentModel]):
        """添加智能体模型"""
        return super().add_model(model)

    def add_environment_config(self, environment_config: Any) -> Self:
        assert is_dataclass(environment_config), "config must be a dataclass"
        self._environment_builder.register(type(environment_config), instance=environment_config, lifespan=Lifespan.singleton)
        return self

    def add_environment(self, environment: IEnvironment | type[IEnvironment]) -> Self:
        match environment:
            case type():
                self._environment_builder.register(IEnvironment, implementation_type=environment, lifespan=Lifespan.singleton)
            case IEnvironment():
                self._environment_builder.register(IEnvironment, instance=environment, lifespan=Lifespan.singleton)
        return self

    def add_replay_buffer(self, buffer: IReplayBuffer | type[IReplayBuffer]) -> Self:
        match buffer:
            case type():
                self.register(IReplayBuffer, implementation_type=buffer, lifespan=Lifespan.singleton)
            case IReplayBuffer():
                self.register(IReplayBuffer, instance=buffer, lifespan=Lifespan.singleton)
        return self

    @property
    def environment(self) -> IEnvironment:
        if self.has_registration(IEnvironment):
            return self.resolve(IEnvironment)
        else:
            environment = self._environment_builder.resolve(IEnvironment).build_environment()
            self.register(IEnvironment, instance=environment)
            return environment

    @property
    def model(self) -> IAgentModel:
        return super().model  # type: ignore

    @property
    def replay_buffer(self) -> IReplayBuffer:
        return self.resolve(IReplayBuffer)

    @property
    def actor_optimizer(self) -> Optimizer:
        optimizer = self.resolve(IOptimizer)
        assert isinstance(optimizer, IAgentOptimizer)
        return optimizer.actor_optimizer

    @property
    def critic_optimizer(self) -> Optimizer:
        optimizer = self.resolve(IOptimizer)
        assert isinstance(optimizer, IAgentOptimizer)
        return optimizer.critic_optimizer

    @property
    def critic_other_optimizer(self) -> Optimizer:
        optimizer = self.resolve(IOptimizer)
        assert isinstance(optimizer, IAgentOptimizer)
        return optimizer.critic_other_optimizer

    @property
    def log_alpha_optimizer(self) -> Optimizer:
        optimizer = self.resolve(IOptimizer)
        assert isinstance(optimizer, IAgentOptimizer)
        return optimizer.log_alpha_optimizer

# TODO other 的 step
    @property
    def actor_scheduler(self) -> LRScheduler:
        scheduler = self.resolve(IScheduler)
        assert isinstance(scheduler, IAgentScheduler)
        return scheduler.actor_scheduler

    @property
    def critic_scheduler(self) -> LRScheduler:
        scheduler = self.resolve(IScheduler)
        assert isinstance(scheduler, IAgentScheduler)
        return scheduler.critic_scheduler

    @property
    def critic_other_scheduler(self) -> LRScheduler:
        scheduler = self.resolve(IScheduler)
        assert isinstance(scheduler, IAgentScheduler)
        return scheduler.critic_other_scheduler

    @property
    def log_alpha_scheduler(self) -> LRScheduler:
        scheduler = self.resolve(IScheduler)
        assert isinstance(scheduler, IAgentScheduler)
        return scheduler.log_alpha_scheduler

    @property
    def loss_function(self) -> IAgentLoss:
        loss_function = super().loss_function
        assert isinstance(loss_function, IAgentLoss)
        return loss_function

    @property
    def train_rewards(self) -> list[float]:
        return self.stats["rewards"]

    @property
    def train_actor_losses(self) -> list[float]:
        return self.stats["actor_losses"]

    @property
    def train_critic_losses(self) -> list[float]:
        return self.stats["critic_losses"]
    # endregion

    def train(self, print_interval: int = 0, method: Literal["behavior_cloning", "ddpg", "sac", "td3"] = "behavior_cloning"):
        self.model.train()
        # 清空 scheduler 计数器
        self.actor_scheduler.last_epoch = -1
        self.critic_scheduler.last_epoch = -1
        self.critic_other_scheduler.last_epoch = -1
        self.log_alpha_scheduler.last_epoch = -1
        match method:

            case "behavior_cloning":
                for epoch in range(self.config.epoch):
                    self.train_single_step_through_behavior_cloning()
                    if print_interval > 0 and epoch % print_interval == 0:
                        print(f"Epoch [{epoch + 1}/{self.config.epoch}], Loss on total dataset: {self.train_losses[-1]:.4f}")
                self.model.soft_update_target_net("actor", tau=1.0)
            case "ddpg" | "sac" | "td3" as offline_method:
                for epoch in range(self.config.epoch):

                    has_train = False
                    while not has_train:
                        has_train = self.train_single_epoch_offline(epoch, method=offline_method)

                    if print_interval > 0 and epoch % print_interval == 0:
                        if len(self.train_actor_losses) > 0:
                            print(f"Epoch [{epoch + 1}/{self.config.epoch}], " +
                                  f"Actor Loss: {self.train_actor_losses[-1]:.4f}, " +
                                  f"Critic Loss: {self.train_critic_losses[-1]:.4f}")

    def train_single_step_through_behavior_cloning(self):

        loss_on_total_dataset = []

        for batch in self.train_dataloader:
            states, actions = self.data_processer.preprocess(batch)
            predicted_actions = self.model(states, None, "action")

            loss = self.loss_function(predicted_actions, actions, "behavior_clone")
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            loss_on_total_dataset.append(loss.item())

        self.actor_scheduler.step()
        self.train_losses.append(mean(loss_on_total_dataset).item())

    def train_through_ddpg(self, epoch: int):
        # 解压数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # region 训练 Critic 网络
        # 计算 t+1 时的 目标 Q 值
        with no_grad():
            q_next = self.model(next_states.cuda(), None, "target_q")
            q_target = rewards.cuda() + self.config.gamma_rl * q_next * (1 - dones.cuda())
        # 计算当前的 Q 值
        q = self.model(states.cuda(), actions.cuda(), "q")

        # 计算 Critic 损失并更新参数 (减小 Q 和 Q_target 的差异)
        critic_loss = self.loss_function(q, q_target, "ddpg_critic")
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # endregion

        # region 训练 Actor 网络
        # 计算当前的 Q 值
        q = self.model(states.cuda(), None, "q")
        # 计算 Actor 损失并更新参数 (最大化Q)
        actor_loss = self.loss_function(q, None, target="ddpg_actor")
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # endregion

        # 记录损失
        self.train_actor_losses.append(actor_loss.item())
        self.train_critic_losses.append(critic_loss.item())

        # 软更新目标网络
        self.model.soft_update_target_net("actor", "critic")

    def train_through_sac(self, epoch: int):

        # 解压数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # region 训练 Critic 网络
        # 计算 t+1 时的 目标 Q 值
        with no_grad():
            q_next = self.model(next_states.cuda(), None, "target_q_sac")
            q_target = rewards.cuda() + self.config.gamma_rl * q_next * (1 - dones.cuda())

        # 计算当前的 Q 值
        q = self.model(states.cuda(), actions.cuda(), "q")
        q_other = self.model(states.cuda(), actions.cuda(), "q_other")

        # 计算 Critic 损失并更新参数 (减小 Q 和 Q_target 的差异)
        critic_loss = self.loss_function(q, q_target, "ddpg_critic")
        critic_other_loss = self.loss_function(q_other, q_target, "ddpg_critic")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_other_optimizer.zero_grad()
        critic_other_loss.backward()
        self.critic_other_optimizer.step()
        # endregion

        # region 训练 Actor 网络

        # 计算当前的 Q 值
        predicted_actions, log_probs = self.model(states.cuda(), None, "action_with_log_prob")
        entropy = -log_probs

        q = self.model(states.cuda(), predicted_actions, "q")
        q_other = self.model(states.cuda(), predicted_actions, "q_other")

        # 计算 Actor 损失并更新参数 (最大化Q)
        actor_loss = self.loss_function(q, None, "sac_actor", q_other=q_other, log_prob=log_probs, log_alpha=self.model.log_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # endregion

        # region 更新 alpha
        alpha_loss = torch.mean(
            (entropy - self.model.config.target_entropy).detach() *
            self.model.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        # endregion

        # 记录损失和累计奖励
        self.train_actor_losses.append(actor_loss.item())
        self.train_critic_losses.append((critic_loss.item() + critic_other_loss.item()) / 2)

        # 软更新目标网络
        self.model.soft_update_target_net("critic", "critic_other")

    def train_through_td3(self, epoch: int, delta: int):

        # 解压数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        with no_grad():
            q_next = self.model(next_states.cuda(), None, "target_q_td3")
            q_target = rewards.cuda() + self.config.gamma_rl * q_next * (1 - dones.cuda())

        q = self.model(states.cuda(), actions.cuda(), "q")
        q_other = self.model(states.cuda(), actions.cuda(), "q_other")

        q_loss = self.loss_function(q, q_target, "ddpg_critic")
        q_other_loss = self.loss_function(q_other, q_target, "ddpg_critic")

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        clip_grad_norm_(self.model.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        self.critic_other_optimizer.zero_grad()
        q_other_loss.backward()
        clip_grad_norm_(self.model.other_critic.parameters(), 1.0)
        self.critic_other_optimizer.step()

        if delta % self.config.policy_delay == 0:
            q = torch.min(self.model(states.cuda(), None, "q"), self.model(states.cuda(), None, "q_other"))
            actor_loss = self.loss_function(q, None, target="ddpg_actor")
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.model.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            self.model.soft_update_target_net("critic", "critic_other", "actor")
            print(f"Actor Loss: {actor_loss.item():.4f}")

        print(f"Critic Loss 1: {q_loss.item():.4f}, Critic Loss 2: {q_other_loss.item():.4f}")

    def train_single_epoch_offline(self, epoch: int, method: str) -> bool:
        # region 单步训练前准备
        state, reward, done, timeout, info = self.environment.reset()  # 重置环境
        total_r = reward  # 累计奖励和标志位
        delta = 0
        # endregion

        while not done and not timeout:

            # 与环境交互
            with no_grad():
                action = self.model(state.cuda(), None, "action")
                next_state, reward, done, timeout, info = self.environment.step(action.cpu())
                self.replay_buffer.append(state, action, reward, next_state, done)
                state = next_state
                total_r += reward

            # 训练
            if not self.replay_buffer.can_sample:
                pass
            else:
                match method:
                    case "ddpg":
                        self.train_through_ddpg(epoch)
                    case "sac":
                        self.train_through_sac(epoch)
                    case "td3":
                        self.train_through_td3(epoch, delta)
                        delta += 1

        # 更新学习率
        match method:
            case "ddpg":
                self.actor_scheduler.step()
                self.critic_scheduler.step()
            case "sac":
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                self.critic_other_scheduler.step()
                self.log_alpha_scheduler.step()
            case "td3":
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                self.critic_other_scheduler.step()
        return True

    @no_grad()
    def evalute_on_environment(self,):
        self.model.eval()
        ob, r, done, timeout, info = self.environment.reset()
        ob_list = []
        done = False
        timeout = False
        while not done and not timeout:
            action = self.model(ob.cuda())
            ob, reward, done, timeout, info = self.environment.step(action.cpu().detach())  # type: ignore
            ob_list.append(ob)
        return ob_list
        ob_list.append(ob)
        return ob_list
