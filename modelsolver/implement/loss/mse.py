from torch import Tensor, stack, tensor
from torch.nn.functional import mse_loss

from modelsolver.abc.functional import ILoss


class LengthWeightedMseLoss(ILoss):
    """长度加权的绝对平方误差损失"""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, labels: Tensor, length: list[int]) -> Tensor:

        losses: list[Tensor] = []  # len = B

        for seq, label, l in zip(x, labels, length):
            mse = mse_loss(seq[1:l], label[1:l])
            losses.append(mse)

        l_weight = (tensor(length) - 1).cuda()  # shape = (B,)
        loss = (stack(losses).cuda() * l_weight / sum(l_weight)).sum()  # 批量平均损失
        return loss.float().cuda()
