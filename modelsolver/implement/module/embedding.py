"""嵌入编码的默认实现."""
from typing import TYPE_CHECKING

import torch
from torch.nn import Conv2d

from modelsolver.abc.model import IModule


class RotationPositionEmbedding(IModule):
    """旋转位置嵌入（RoPE）模块.

    参考: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """旋转位置嵌入, 输出形状与输入形状一致"""
            ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, dim)，对每个位置施加旋转。
        seq_len = x.size(-2)
        sin, cos = self.rope(self.hidden_size, seq_len, x.device, x.dtype)
        return x * cos + self.rotate_half(x) * sin



    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """把最后一维拆成 (D/2, 2)，旋转半平面：(-x2, x1)。"""
        D = x.size(-1)
        x = x.reshape(*x.shape[:-1], D // 2, 2)
        x1, x2 = x[..., 0], x[..., 1]
        return torch.cat([-x2, x1], dim=-1)

    def rope(
        self,
        dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """预计算 1D RoPE 的 sin/cos 基。

        频率按 $\theta_j = 10000^{-2j/dim}$ 生成，返回 (1, 1, seq_len, dim) 的 sin/cos，
        便于与 (B, H, N, D) 的 q/k 直接广播。
        """
        half = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, half)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        sin = torch.sin(emb).reshape(1, 1, seq_len, dim)
        cos = torch.cos(emb).reshape(1, 1, seq_len, dim)
        return sin.to(dtype), cos.to(dtype)

class PatchEmbedding(IModule):
    """用卷积把图像切成 patch 并映射到隐维度的编码器."""

    def __init__(self, input_size: int, patch_size: int, in_channels: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) -> (B, C, H, W) -> conv -> (B, T, hidden)
        B, H, W, C = x.shape
        assert C < 7, f"疑似缺少 transpose，得到 x.shape={x.shape}"
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.proj(x)  # (B, hidden, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, T, hidden)
        return x



    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """通过卷积操作将图像输入切分为 patch 并映射到隐藏维度的编码器

            Args:
                x (torch.Tensor): 图像输入. shape = (B, H, W, C)

            Returns:
                编码输出 (torch.Tensor): shape = (B, T, hidden). 其中 T = (H/p)*(W/p), hidden = 隐维度数
            """
            ...