"""专家数据索引与采样模块"""
import pandas as pd
import torch
from torch import Tensor, from_numpy, no_grad


class ExpertStore:
    """
    仅负责:
    1) 接收 DataFrame(state/action)
    2) 构建tensor缓存
    3) 按当前state_part做top-k最近邻goal采样
    """

    def __init__(self,
                 state_info:tuple[str, int],
                 action_info:tuple[str, int],
                 df: pd.DataFrame) -> None:
        self.state_info = state_info
        self.action_info = action_info
        self.goal_dim = self.state_info[1] // 2
        self.expert_state, self._actions = self._init_expert_data_from_dataframe(df)

    def _init_expert_data_from_dataframe(self, df:pd.DataFrame) -> tuple[Tensor, Tensor]:
        assert len(df) > 0, "expert dataframe 不能为空"
        assert self.state_info[0] in df.columns and self.action_info[0] in df.columns, "expert dataframe 不包含指定的列名"

        goal_states = []
        actions = []

        for _, row in df.iterrows():
            s = from_numpy(row[self.state_info[0]]).reshape(-1).float()
            a = from_numpy(row[self.action_info[0]]).reshape(-1).float()
            goal_states.append(s[:self.goal_dim])
            actions.append(a)

        return torch.stack(goal_states, dim=0), torch.stack(actions, dim=0)  # shape = (N, S/2), (N, A)

    @no_grad
    def sample_expert_states(self, batch_state: Tensor, topk: int) -> Tensor:
        """
        对batch中每个当前state:
        - 找到expert中最近的top-k
        - 在top-k中随机选1个goal


        Args:
            batch_state (Tensor): 传入的状态，shape = (B, S/2)，**注意: 这里的state应该是不包含goal信息的完整状态的一部分，即obs_dim部分**
            topk (int): 每个输入状态对应的最近邻专家状态数量.

        Returns:
            expert_state (Tensor): 采样到的专家状态, shape = (B, S/2)
        """
        expert = self.expert_state  # type: ignore
        k = max(1, min(int(topk), expert.shape[0]))


        dist = torch.cdist(batch_state, self.expert_state, p=2)  # (B, N)
        # 最近k个的索引, shape = (B, k)
        _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)

        bsz = batch_state.shape[0]
        pick = torch.randint(0, k, (bsz,), dtype=torch.long)
        row = torch.arange(bsz, dtype=torch.long)
        chosen = nn_idx[row, pick]

        return self.expert_state[chosen].clone()