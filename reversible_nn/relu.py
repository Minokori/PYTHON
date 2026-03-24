import torch

from modelsolver.abc.model import IActivateFunction


class PReLU(IActivateFunction):
    """误差很大, 不要堆叠太多"""

    def __init__(self):
        super().__init__()

        self.weight_of_prelu = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_of_prelu is None:
            self.weight_of_prelu = torch.nn.Parameter(torch.randn(1)).clamp(1e-1).to(x.device)
        else:
            self.weight_of_prelu = self.weight_of_prelu.clamp(1e-1)
        # self.weight_of_prelu = self.weight_of_prelu.clamp(1e-1) or torch.nn.Parameter(torch.randn(1)).clamp(1e-1)
        return torch.nn.functional.prelu(x.permute(0, 2, 1), self.weight_of_prelu).permute(0, 2, 1)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        z = torch.empty_like(x)
        mask = x >= 0
        z[mask] = x[mask]
        z[~mask] = x[~mask] / self.weight_of_prelu  # type: ignore
        return z.permute(0, 2, 1)
