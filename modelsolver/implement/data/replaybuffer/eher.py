"""融合HER和专家重标注的经验回放池"""

from dataclasses import dataclass

import pandas as pd
import torch
from dataclasses_json import dataclass_json
from torch import Tensor, no_grad

from modelsolver.abc.config import ReplayBufferConfig
from modelsolver.abc.data import IReplayBuffer
from modelsolver.abc.reward import IReward
from modelsolver.implement.data.replaybuffer.store.expert import ExpertStore
from modelsolver.implement.data.replaybuffer.store.trajectory import \
    TrajectoryStore


@dataclass_json
@dataclass
class EHERConfig(ReplayBufferConfig):
    """HER经验回放池配置"""
    her_p: float = 0.8
    """HER采样概率"""

    expert_data_path: str = ""
    """pkl 格式的专家数据路径, 数据应包含'state'和'action'两列"""

    use_expert_goal: bool = True
    """是否启用专家goal重标注"""

    expert_p: float = 0.5
    """在HER命中的样本中, 使用专家goal的概率"""

    expert_topk: int = 32
    """按当前state找最近邻后, 在top-k中随机采样"""

    max_expert_goal_dist: float = -1.0
    """专家goal距离过滤阈值(L2). <0表示不启用"""

    expert_state_col: str = "state"
    expert_action_col: str = "action"

class ExpertHEReplayBuffer(IReplayBuffer):
    """
    仅负责编排:
    - 基础随机采样
    - HER索引选择
    - future/expert重标注
    - reward/done重算
    """

    def __init__(self, config: ReplayBufferConfig, reward: IReward):
        assert issubclass(type(config), EHERConfig), "HER 经验回放池需要 HEReplayBufferConfig 实例作为配置"

        self._config = config
        self._her_reward_fn = reward

        self._g_dim = int(self._config.state_dim // 2)

        self._store = TrajectoryStore(
            capacity=int(self._config.capacity),
            state_dim=int(self._config.state_dim),
            action_dim=int(self._config.action_dim),
        )

        self._expert_index = ExpertStore(
            state_info=(self.config.expert_state_col, self.config.state_dim),
            action_info=(self.config.expert_action_col, self.config.action_dim),
            df=pd.read_pickle(self.config.expert_data_path) )

    @property
    def config(self) -> EHERConfig:
        return self._config  # type: ignore

    def __len__(self) -> int:
        return self._store.size

    @property
    def can_sample(self) -> bool:
        return self._store.size > self._config.minimal_capacity


    def append(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        new: bool = False
    ) -> None:
        assert state.is_cpu and action.is_cpu and reward.is_cpu and next_state.is_cpu and done.is_cpu, \
            "输入的 Tensor 必须在 CPU 上"

        if new:
            self._store.start_new_trajectory()

        self._store.append_transition(state, action, reward, next_state, done)

    @no_grad
    def sample(self, her: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = int(self.config.batch_size)
        g_index = self._g_dim

        valid_pos = self._store.get_valid_positions()
        n_valid = valid_pos.numel()
        assert n_valid > 0, "buffer has no valid samples"

        # 基础随机抽样
        pos_now = valid_pos[torch.randint(0, n_valid, (batch_size,))]
        states_t, actions_t, rewards_t, next_states_t, dones_t = self._store[pos_now]


        # HER索引抽取
        her_idx = torch.nonzero(torch.rand(batch_size) < self.config.her_p, as_tuple=False).flatten()  # shape = (her, )


        # 不采用HER或不满足HER条件时, 直接返回原始采样结果
        if her_idx.numel() == 0 or not her:
            return states_t, actions_t, rewards_t, next_states_t, dones_t

        # HER子集拆分为 expert / future
        idx_e, idx_f = self._split_her_idx(her_idx)

        # --- future relabel ---
        goal_pos = self._store.sample_future_goal_positions(pos_now[idx_f])
        goal_f = self._store.get_states_by_positions(goal_pos)[:, :g_index]

        states_t[idx_f, g_index:] = goal_f
        next_states_t[idx_f, g_index:] = goal_f

        # --- expert relabel ---
        current_state_part = states_t[idx_e, :g_index]

        goal_e = self._expert_index.sample_expert_states(current_state_part,self.config.expert_topk,)

        #  阈值过滤: 仅保留距离当前状态较近的专家goal
        if self.config.max_expert_goal_dist > 0:
            dist = torch.norm(current_state_part - goal_e, dim=1)
            ok = dist <= self.config.max_expert_goal_dist
            if ok.any():
                idx_e = idx_e[ok]
                goal_e = goal_e[ok]
            else:
                idx_e = torch.empty((0,), dtype=torch.long)

        if idx_e.numel() > 0:
            states_t[idx_e, g_index:] = goal_e
            next_states_t[idx_e, g_index:] = goal_e

        # 重算HER命中样本的reward/done
        rew_new, done_new = self._her_reward_fn(
            state=states_t[her_idx],
            action=actions_t[her_idx],
            next_state=next_states_t[her_idx],
        )
        rewards_t[her_idx] = rew_new
        dones_t[her_idx] = done_new

        return states_t, actions_t, rewards_t, next_states_t, dones_t


    def _split_her_idx(self, her_idx: Tensor) -> tuple[Tensor, Tensor]:
        if self.config.use_expert_goal:
            mask_expert = torch.rand(her_idx.numel()) < self.config.expert_p
            expert_local = torch.nonzero(mask_expert, as_tuple=False).flatten()
            future_local = torch.nonzero(~mask_expert, as_tuple=False).flatten()
        else:
            expert_local = torch.empty((0,), dtype=torch.long)
            future_local = torch.arange(her_idx.numel(), dtype=torch.long)
        return her_idx[expert_local], her_idx[future_local]