"""功能函数"""
import torch


def get_device_and_dtype(x: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    return x.device, x.dtype