"""预实现的数据集类"""

from dataclasses import dataclass, field
from typing import Literal, Self

import pandas as pd
from dataclasses_json import dataclass_json

from modelsolver.abc.config import DataConfig
from modelsolver.abc.data import IDataset


@dataclass_json
@dataclass
class PandasDataConfig(DataConfig):
    """使用 Pandas DataFrame 作为数据集的配置类"""
    pickle_file_path:str = ""
    """数据集的 pickle 文件路径"""
    sample_columns: list[str] = field(default_factory=list)
    """样本列的名称列表"""
    label_columns: list[str] = field(default_factory=list)
    """标签列的名称列表"""
    chunk_size: int = 0
    """一条序列裁剪到的长度, 一条序列可能因此裁剪为若干条数据"""
    chunk_num: int = 1
    """一条序列裁剪到的个数.
    仅当 chunk_mode 为 "random" 时有效.
    """
    chunk_mode: Literal["random", "sequential"] = "sequential"

# TODO config:DataConfig 的子类, 用于 PandasDataset
class PandasDataset(IDataset):
    """使用 Pandas DataFrame 作为数据集的基础类

    有关 DataFrame :
    + 每一行是一个样本
    + 样本的特征列和标签列由 `sample_columns` 和 `label_columns` 在 config 中指定
    + iter 时, 返回样本的 DataFrame, 而非 Series (*即使只有一行数据*)
    + *若 DataFrame 的数据需要预处理以使用, 请在自定义 IDataProcessor 类中实现相关逻辑*
    + *一般而言, 不需要继承 PandasDataset, 使用 IDataProcessor*
    """
    def __init__(self, config: PandasDataConfig):
        self._dataframe = pd.read_pickle(config.pickle_file_path)

    def __getitem__(self, index: int | list[int] | slice) -> pd.DataFrame:
        """单行数据同样以 `Dataframe` 返回, 而非 `Series`"""
        match index:
            case int():
                index = [index]
            case slice() | list():
                pass

        return self._dataframe.iloc[index, :]

    def __len__(self) -> int:
        return len(self._dataframe)

    def __add__(self, other: Self) -> Self:
        self._dataframe = pd.concat([self._dataframe, other._dataframe], ignore_index=True)
        return self