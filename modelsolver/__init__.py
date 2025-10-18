"""采用依赖注入 (DI) 的模型训练一键式解决方案"""

# region 库导入
import os
from dataclasses import is_dataclass
from math import floor, log
from typing import Any

from clean_ioc import Container, DependencySettings, Lifespan
from clean_ioc.registration_filters import with_name
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy import float32
from numpy.typing import NDArray
from torch import load, no_grad, save
from torch.autograd import set_detect_anomaly
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import Dataset, random_split

from modelsolver import abc
from modelsolver.abc.config import DataConfig, HyperParameterConfig
from modelsolver.abc.data import IDataLoader, IDataProcesser, IDataset
from modelsolver.abc.functional import ILoss, IOptimizer, IScheduler
from modelsolver.abc.model import IModel


# endregion

__all__ = ["ModelSolver", "abc"]

# TODO model的 __call__ 方法的 **kwargs


class ModelSolver(Container):
    """模型解决方案"""

    def __init__(self):
        super().__init__()
        self.register(FontProperties, instance=FontProperties("./data/SourceHanSansLite.ttf", size=14))

        # 训练过程中的损失记录
        self.stats = self.init_stats()

        # 子 ioc 容器, 构建模型和数据加载器
        self._model_builder = Container()
        self._dataloader_builder = Container()

        # dataloader 缓存
        self._dataloaders: list[IDataLoader] = []

    # region 注册损失函数, 优化器, 学习率调度器, 配置项, 数据处理器, 使 ioc 更易用
    # TODO 具名loss, optimizer, scheduler, 因为RL有可能会有多个
    def add_loss_function(self, loss_function: type[ILoss]):
        """注册损失函数"""
        self.register(ILoss, loss_function, lifespan=Lifespan.singleton)
        return self

    def add_optimizer(self, optimizer: type[IOptimizer]):
        """注册优化器"""
        self.register(IOptimizer, implementation_type=optimizer, lifespan=Lifespan.singleton)
        return self

    def add_lr_scheduler(self, lr_scheduler: type[IScheduler]):
        """注册学习率调度器"""
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

    def add_model_component(self, module: type, implementation: type, name: str, singleton: bool = False):
        """注册模型组件, 通过 ioc 容器构建模型实例"""
        self._model_builder.register(module,
                                     implementation,
                                     lifespan=Lifespan.singleton if singleton else Lifespan.transient, dependency_config={
                                         "config": DependencySettings(
                                             filter=with_name(name))})
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
        return self.resolve(IOptimizer).optimizer

    @property
    def scheduler(self):
        return self.resolve(IScheduler).scheduler

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

    def init_stats(self) -> dict[str, list[float]]:
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
                train_loss = self.train_single_epoch(epoch)
                test_loss, other_stats = self._evaluate()

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

                    y_hat: dict[str, NDArray[float32]] = self.data_processer.postprocess(label_hat, target="label")
                    y: dict[str, NDArray[float32]] = self.data_processer.postprocess(label, target="label")

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
    # endregion
    # endregion
