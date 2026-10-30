"""注意力模块"""

from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import LayerNorm
from torch.nn.functional import scaled_dot_product_attention

from modelsolver.abc.model import IModule
from modelsolver.implement.module.embedding import RotationPositionEmbedding
from modelsolver.implement.module.linear import Linear
from modelsolver.types import EMPTY, CacheArg


class Attention(IModule):
    """多头注意力(支持可选 RoPE)"""
    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
    def __init__(self, dim: int, rope: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = max(1, dim // 64)
        # 默认 64 维一组;当 dim < 64 时退化为单头,head_dim=dim.
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = Linear(dim, dim * 3)
        self.proj = Linear(dim, dim)
        self.rope = rope
        if rope:
            self.pe = RotationPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # 每个都是 (B, N, H, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # (B, H, N, D)

        if self.rope:
            q = self.pe(q)
            k = self.pe(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        x = attn @ v  # (B, H, N, D)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class ConditionalAttention(IModule):
    """条件注意力模块

    将条件 token ***c*** 拼接到输入序列 *x* 的末尾,然后对整个序列进行注意力计算.

    头数按 64 维一组划分, 按照 DiT/TARFlow 的默认配置.

    qkv 线性层和输出投影线性层使用 xavier_uniform 初始化, 偏置初始化为零.
    """
    if TYPE_CHECKING:
        def __call__(
            self,
            x: torch.Tensor,

            c: torch.Tensor,
            k_cache: CacheArg = EMPTY,
            v_cache: CacheArg = EMPTY,
            mask: torch.Tensor | None = None,
            which_cache: Literal["cond", "uncond"] = "cond",
            train: bool = True,
        ) -> tuple[torch.Tensor, CacheArg, CacheArg]: ...

    def __init__(self, channels:int):
        super().__init__()
        self.channels:int = channels
        """通道数"""
        self.num_heads:int = max(1, channels // 64)
        """注意力头数"""
        self.head_dim:int =channels // self.num_heads
        """每个注意力头的维度"""
        assert self.head_dim * self.num_heads ==channels, (
            "in_channels must be divisible by num_heads"
        )
        self.qkv:Linear = Linear(channels, 3 * channels).init_weights("xavier_uniform").init_bias("zeros")
        """QKV 线性层"""
        self.proj:Linear = Linear(channels, channels) .init_weights("xavier_uniform").init_bias("zeros")
        """输出投影线性层"""
        self.norm:LayerNorm = LayerNorm(channels, elementwise_affine=False)
        """LayerNorm 层"""


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        c: torch.Tensor,
        k_cache: CacheArg = EMPTY,
        v_cache: CacheArg = EMPTY,
        which_cache: Literal["cond", "uncond"] = "cond",
        train: bool = True,
    ) -> tuple[torch.Tensor, CacheArg, CacheArg]:
        # x: (B, T, C);c: (B, 1, C).
        B, T, C = x.shape
        Tc = c.shape[1]
        assert c.shape == (B, Tc, C), "c must be (B, 1, C)"

        # 拼接 [x; c],再做 qkv 线性变换.
        x = torch.cat([x, c], dim=1)  # (B, T+Tc, C)
        x = self.norm(x)
        qkv = self.qkv(x)  # (B, T+Tc, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, T + Tc, self.num_heads, self.head_dim)
        k = k.reshape(B, T + Tc, self.num_heads, self.head_dim)
        v = v.reshape(B, T + Tc, self.num_heads, self.head_dim)
        # 查询只来自原序列,键值来自 [原序列 + 条件 token].
        q = q[:, :-Tc]

        if train:
            assert k_cache is EMPTY and v_cache is EMPTY, "train 阶段不应使用 KV-cache"
        else:
            assert k_cache is not EMPTY and v_cache is not EMPTY, "推理阶段必须提供 KV-cache"
            # 推理:用 KV-cache 缓存历史 token 的键值,逐 token 解码.
            k_c, v_c = k[:, -Tc:], v[:, -Tc:]
            k, v = k[:, :-Tc], v[:, :-Tc]
            idx = k_cache["idx"]
            k_cache[which_cache][:, idx : idx + 1] = k
            v_cache[which_cache][:, idx : idx + 1] = v
            k = k_cache[which_cache]
            v = v_cache[which_cache]
            k = torch.cat([k, k_c], dim=1)  # 追加条件 token 的 key
            v = torch.cat([v, v_c], dim=1)

        # 掩码右侧补 True,让条件 token 始终可见(非因果).
        mask = torch.cat([mask, mask.new_ones(mask.size(0), Tc)], dim=1)  # (Q, T+Tc)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Q, T+Tc)

        out = ConditionalAttention.attention(q, k, v, mask=mask).reshape(B, T, C)
        out = self.proj(out)
        return out, k_cache, v_cache

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """缩放点积注意力:softmax(QK^T/√d) V.

        q/k/v 形状:(B, Q, H, D) / (B, K, H, D) / (B, K, H, D).
        mask 形状:(..., Q, K),True 表示允许注意力.
        """
        # SDPA 要求 (B, H, L, D),所以先把 H 维换到前面
        # q: (B, Q, H, D) -> (B, H, Q, D)
        # k: (B, K, H, D) -> (B, H, K, D)
        # v: (B, K, H, D) -> (B, H, K, D)
        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)

        if mask is not None:
            # SDPA 的 attn_mask 语义:True 表示"参与注意力"(允许注意力),False 表示"屏蔽".
            # 但融合内核(Flash / Memory-Efficient)对 bool mask 的支持有限,
            # 需要保证 mask 能广播到 (B, H, Q, K) 且 dtype 为 bool.
            # 如果 mask 形状为 (B, 1, Q, K) 或 (B, H, Q, K),可直接使用.
            attn_mask = mask
        else:
            attn_mask = None

        # PyTorch 2.x 中 scale 默认即为 1/sqrt(E),无需手动传入;
        # 若想显式表达,可传 scale=head_dim ** -0.5.
        out = scaled_dot_product_attention(
            q_, k_, v_,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        # out: (B, H, Q, D) -> (B, Q, H, D)
        return out.transpose(1, 2)