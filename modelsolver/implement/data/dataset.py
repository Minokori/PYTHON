"""预实现的数据集类"""

from dataclasses import field
from typing import Self

import pandas as pd

from modelsolver.abc.config import DataConfig
from modelsolver.abc.data import IDataset


class PandasDataConfig(DataConfig):
    """使用 Pandas DataFrame 作为数据集的配置类"""
    pickle_file_path:str
    """数据集的 pickle 文件路径"""
    sample_columns: list[str] = field(default_factory=list)
    """样本列的名称列表"""
    label_columns: list[str] = field(default_factory=list)
    """标签列的名称列表"""

# TODO config:DataConfig 的子类, 用于 PandasDataset
class PandasDataset(IDataset):
    """使用 Pandas DataFrame 作为数据集的基础类"""
    def __init__(self, config: DataConfig):
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