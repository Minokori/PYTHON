from typing import TYPE_CHECKING

from torch import Tensor
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
