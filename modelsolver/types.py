"""一些类型定义."""
from enum import Enum, auto
from typing import TypedDict, TypeAlias

from torch import Tensor


class Cache(TypedDict):
    """条件注意力计算时使用的kv缓存."""
    cond:Tensor
    """有条件引导时使用的缓存"""
    uncond:Tensor
    """无条件引导时使用的缓存"""
    idx:int
    """当前缓存的索引,用于在推理阶段逐步更新缓存"""


class _EmptyCache(Enum):
    """无缓存状态的哨兵类型。"""

    EMPTY = auto()


EMPTY = _EmptyCache.EMPTY

CacheArg: TypeAlias = Cache | _EmptyCache