"""专家数据索引与采样模块"""

import pandas as pd
import torch
from torch import Tensor, from_numpy, no_grad

from modelsolver.abc.config import ReplayBufferConfig
from modelsolver.abc.data import IExpertStore
from modelsolver.abc.reward import IReward
from modelsolver.implement.data.replaybuffer.eher import EHERConfig


class ExpertStore(IExpertStore):
    """默认的专家数据存储模块"""
    _expert_state: Tensor
    """专家状态缓存, shape = (N, S/2)"""
    _expert_action: Tensor
    """专家动作缓存, shape = (N, A)"""

    def __init__(self, config:ReplayBufferConfig, reward_fn:IReward) -> None:
        assert issubclass(type(config), EHERConfig), "IExpertStore 需要 HEReplayBufferConfig 实例作为配置"
        self._config = config
        self.state_info = self.config.expert_state_col
        self.action_info = self.config.expert_action_col
        self.goal_dim = self.config.state_dim // 2
        self._reward_fn = reward_fn
        self._init_expert_data_from_dataframe(pd.read_pickle(self.config.expert_data_path))


    @property
    def config(self) -> EHERConfig:
        return self._config  # type: ignore

    @no_grad
    def sample_expert_states(self, batch_state: Tensor, topk: int) -> tuple[Tensor, Tensor]:
        # TODO : 应该先:
        # 1) 筛选出专家数据中 reward 更优的专家goal (reward 可预先计算, 存储在缓存中, 速度较快)
        # 2) 再在剩余的专家goal中做最近邻top-k
        # 3) 返回最终选中的专家goal
        # 4) 每一步都需要考虑: 万一没有专家goal满足条件, 该怎么办
        k = max(1, min(topk, self._expert_state.shape[0]))


        dist = torch.cdist(batch_state[:,:self.goal_dim], self._expert_state[:,:self.goal_dim], p=2)  # (B, N)
        # 最近k个的索引, shape = (B, k)
        _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)

        bsz = batch_state.shape[0]
        pick = torch.randint(0, k, (bsz,), dtype=torch.long)
        row = torch.arange(bsz, dtype=torch.long)
        chosen = nn_idx[row, pick]


        chosen_final = self._select_valid_expert_goals(batch_state, self._expert_state[chosen], chosen)

        return self._expert_state[chosen].clone(),chosen


    # region private methods
    def _init_expert_data_from_dataframe(self, df:pd.DataFrame) -> None:
        assert len(df) > 0, "expert dataframe 不能为空"
        assert self.state_info[0] in df.columns and self.action_info[0] in df.columns, "expert dataframe 不包含指定的列名"

        goal_states = []
        actions = []

        for _, row in df.iterrows():
            s = from_numpy(row[self.state_info[0]]).reshape(-1).float()
            a = from_numpy(row[self.action_info[0]]).reshape(-1).float()
            goal_states.append(s)
            actions.append(a)

        self._expert_state=torch.stack(goal_states, dim=0)  # shape = (N, S/2)
        self._expert_action=torch.stack(actions, dim=0)  # shape = (N, A)
    def _select_valid_expert_goals(self, ob: Tensor, goals_ob: Tensor, chosen_indices: Tensor) -> Tensor:
        """选择哪些专家goal是合适的(距离当前状态较近)

        Args:
            ob (Tensor): 传入的 ob
            goals_ob (Tensor): 对应的专家 ob
            chosen_indices (Tensor): 被选中的专家状态的索引, shape = (B,)

        Returns:
            是否合理 (Tensor): shape = (N_ob,1), bool
        """
        if self.config.max_expert_goal_dist >0:
            dist = torch.norm(ob[:, :self.config.goal_index] - goals_ob, dim=1)
            ok = dist <= self.config.max_expert_goal_dist
        else:
            ok = torch.ones((goals_ob.shape[0],), dtype=torch.bool)

        ob_reward,_ = self._reward_fn(next_state=goals_ob)
        e_reward = self._reward_fn(next_state=self._expert_state[chosen_indices])[0]

        ok2 = (ob_reward <= e_reward + 1e-3).reshape(-1)  # 仅当专家reward更优时才使用专家goal

        ok_final = ok & ok2

        # ok : 合适的专家goal布尔索引 (在 goals_e 里的索引), shape = (num_expert_goals,)
        return ok_final
    # endregion