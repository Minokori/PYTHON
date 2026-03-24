from typing import TYPE_CHECKING

from torch import Tensor
import torch
from torchmetrics import RelativeSquaredError

from modelsolver.abc.functional import ILoss


class RelativeSquaredErrorLoss(ILoss):
    """相对均方误差"""

    def __init__(self) -> None:
        super().__init__()
        self.rse = None

    if TYPE_CHECKING:
        def __call__(self, predicted: Tensor, label: Tensor): ...

    def forward(self, predicted: Tensor, labels: Tensor):
        B, *_, C = predicted.shape
        if not self.rse:
            self.rse = RelativeSquaredError(C, True).to(predicted)
        return self.rse(predicted, labels)

class LengthWeightedRseLoss(ILoss):
    """长度加权的相对平方误差损失"""
    def __init__(self):
        super().__init__()
        self.rse = None
        self.dirichlet = None
        self.normal = None

    def forward(self, x: torch.Tensor, labels: torch.Tensor, length: list[int]) -> torch.Tensor:
        B, L, C = x.shape
        F = 0.2
        assert x.shape == labels.shape
        assert len(length) == B
        # 懒加载
        self.rse = self.rse or RelativeSquaredError(C, True).to(x.device)
        self.dirichlet = self.dirichlet or torch.distributions.Dirichlet(torch.ones(C).to(x.device))
        # 每个通道随机权重
        # rand_weight = self._get_rand_weight(labels)
        normal_weight = self._get_normal_weight(labels)
        # 和0有关的权重
        unzero_weight = self._get_unzero_weight(labels, length)

        # 混合权重
        hybrid_weight = (1 - F) * normal_weight + F * unzero_weight

        x = x * hybrid_weight
        labels = labels * hybrid_weight

        losses: list[torch.Tensor] = []  # len = B

        for seq, label, l in zip(x, labels, length):
            rse = self.rse(seq[1:l], label[1:l])
            losses.append(rse)

        l_weight = (torch.tensor(length) - 1).cuda()  # shape = (B,)
        loss = (torch.stack(losses).cuda() * l_weight / torch.sum(l_weight)).sum()  # 批量平均损失
        return loss.float().cuda()

    @torch.no_grad()
    def _get_unzero_weight(self, labels: torch.Tensor, length: list[int]) -> torch.Tensor:
        B, L, C = labels.shape
        unzero_count = torch.zeros(C, device=labels.device)
        for label, l in zip(labels, length):
            true_label = label[0:l]  # (L,C)
            unzero_count += true_label[true_label.abs() > 1e-6].sum(dim=0) / l   # shape = (C,)
        unzero_count = unzero_count / B
        return unzero_count / unzero_count.sum()

    @torch.no_grad()
    def _get_normal_weight(self, labels: torch.Tensor) -> torch.Tensor:
        B, L, C = labels.shape
        self.normal = self.normal or torch.distributions.Normal(0, 1)

        weight = self.normal.sample((C,))
        weight = torch.softmax(weight, dim=0)
        return weight.to(labels)

    @torch.no_grad()
    def _get_rand_weight(self, labels: torch.Tensor) -> torch.Tensor:
        B, L, C = labels.shape
        self.dirichlet = self.dirichlet or torch.distributions.Dirichlet(torch.ones(C).to(labels.device))
        return self.dirichlet.sample()