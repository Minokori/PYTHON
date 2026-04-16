from dataclasses import dataclass

import numpy as np
import torch
from dataclasses_json import dataclass_json
from torch import Tensor, no_grad

from modelsolver.abc.config import ReplayBufferConfig
from modelsolver.abc.data import IReplayBuffer
from modelsolver.abc.reward import IReward


@dataclass_json
@dataclass
class HEReplayBufferConfig(ReplayBufferConfig):
    """HER经验回放池配置"""
    her_p: float = 0.8
    """HER采样概率"""
    
    # 新增：专家目标重标注
    use_expert_goal: bool = True
    """是否启用专家目标重标注"""

    expert_p: float = 0.5
    """在HER命中的样本中, 使用专家goal的概率"""

    # 新增：针对性抽取策略
    expert_topk: int = 32
    """从最近邻中取top-k后再随机采样"""

    max_expert_goal_dist: float = -1.0
    """专家goal距离过滤阈值(基于state_part的L2距离). <0表示不启用过滤"""


class SimpleHEReplay(IReplayBuffer):
    """高性能版简单HER实现（扁平存储 + 可注入奖励函数 + 基于DataFrame的专家goal重标注）"""

    def __init__(self, config: ReplayBufferConfig, reward: IReward):
        assert issubclass(type(config), HEReplayBufferConfig), "HER 经验回放池需要 HEReplayBufferConfig 实例作为配置"

        self._config = config
        self._her_reward_fn = reward

        self._capacity = int(self._config.capacity)
        self._size = 0
        self._position_to_write = 0

        self._states = torch.empty((self.config.capacity, self.config.state_dim), dtype=torch.float32, device="cpu")
        self._actions = torch.empty((self.config.capacity, self.config.action_dim), dtype=torch.float32, device="cpu")
        self._rewards = torch.empty((self.config.capacity, 1), dtype=torch.float32, device="cpu")
        self._next_states = torch.empty((self.config.capacity, self.config.state_dim), dtype=torch.float32, device="cpu")
        self._dones = torch.empty((self.config.capacity, 1), dtype=torch.float32, device="cpu")

        # 轨迹信息
        self._trajectory_id_table = torch.full((self._capacity,), -1, dtype=torch.long)
        self._trajectory_step_table = torch.zeros((self._capacity,), dtype=torch.long)
        self._trajectory_length_table = torch.zeros((self._capacity,), dtype=torch.long)

        self._trajectory_counter = 0
        self._current_trajectory_id = -1
        self._current_trajectory_start_position = -1
        self._current_trajectory_length = 0

        # 可采样位置缓存
        self._valid_pos_cache: Tensor | None = None
        self._cache_dirty = True

        # 状态切分维度
        self._g_dim = self.config.state_dim // 2

        # 你提供的专家数据DataFrame（外部赋值）
        # 约定列: "state", "action"
        self._expert_data = None

        # 为加速最近邻查询缓存专家state/action张量
        self._expert_states_tensor: Tensor | None = None  # shape=(N, g_dim)
        self._expert_actions_tensor: Tensor | None = None # shape=(N, action_dim)
        self._expert_cache_dirty: bool = True

    @property
    def config(self) -> HEReplayBufferConfig:
        return self._config  # type: ignore

    def __len__(self) -> int:
        return self._size

    @property
    def can_sample(self) -> bool:
        return self._size > self._config.minimal_capacity

    # -----------------------------
    # 对外接口：设置/更新专家DataFrame
    # -----------------------------
    def set_expert_data(self, expert_df) -> None:
        """
        设置专家数据DataFrame, 列要求包含:
        - state
        - action
        """
        if expert_df is None:
            self._expert_data = None
            self._expert_states_tensor = None
            self._expert_actions_tensor = None
            self._expert_cache_dirty = True
            return

        assert "state" in expert_df.columns and "action" in expert_df.columns, \
            "expert_df 必须包含列: 'state' 和 'action'"

        self._expert_data = expert_df
        self._expert_cache_dirty = True
        self._rebuild_expert_cache_if_needed()

    def append(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        new: bool = False
    ) -> None:
        """向池中添加一条转移 `(s,a,r,s',done)`"""
        assert state.is_cpu and action.is_cpu and reward.is_cpu and next_state.is_cpu and done.is_cpu, \
            "输入的 Tensor 必须在 CPU 上"

        if new or self._current_trajectory_id < 0:
            self._start_a_new_trajectory()
        self._write_a_transition(state, action, reward, next_state, done)

    @no_grad
    def sample(self, her: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = int(self.config.batch_size)
        g_index = self.config.state_dim // 2

        valid_pos = self._get_valid_positions()
        n_valid = valid_pos.numel()
        assert n_valid > 0, "buffer has no valid samples"

        pos_now = valid_pos[torch.randint(0, n_valid, (batch_size,))]

        states_t = self._states[pos_now].clone()
        actions_t = self._actions[pos_now].clone()
        rewards_t = self._rewards[pos_now].clone()
        next_states_t = self._next_states[pos_now].clone()
        dones_t = self._dones[pos_now].clone().float()

        if not her:
            if dones_t.ndim == 1:
                dones_t = dones_t.unsqueeze(-1)
            return states_t, actions_t, rewards_t, next_states_t, dones_t

        her_idx = torch.nonzero(torch.rand(batch_size) < self.config.her_p, as_tuple=False).flatten()
        if her_idx.numel() == 0:
            if dones_t.ndim == 1:
                dones_t = dones_t.unsqueeze(-1)
            return states_t, actions_t, rewards_t, next_states_t, dones_t

        # 在HER样本中划分 expert / future
        self._rebuild_expert_cache_if_needed()
        has_expert = (
            self.config.use_expert_goal
            and self._expert_states_tensor is not None
            and self._expert_states_tensor.shape[0] > 0
        )

        if has_expert:
            mask_expert = torch.rand(her_idx.numel()) < self.config.expert_p
            expert_local = torch.nonzero(mask_expert, as_tuple=False).flatten()
            future_local = torch.nonzero(~mask_expert, as_tuple=False).flatten()
        else:
            expert_local = torch.empty((0,), dtype=torch.long)
            future_local = torch.arange(her_idx.numel(), dtype=torch.long)

        # ---------- future HER ----------
        if future_local.numel() > 0:
            idx_f = her_idx[future_local]
            pos_h = pos_now[idx_f]
            step_h = self._trajectory_step_table[pos_h]
            ep_len_h = self._trajectory_length_table[pos_h]

            max_off = (ep_len_h - step_h - 1).clamp_min(0)
            u = torch.rand_like(max_off, dtype=torch.float)
            off = (u * (max_off.to(torch.float) + 1.0)).to(torch.long)

            goal_step = step_h + off
            ep_start = (pos_h - step_h) % self._capacity
            pos_goal = (ep_start + goal_step) % self._capacity

            goal_f = self._states[pos_goal][:, :g_index]
            states_t[idx_f, g_index:] = goal_f
            next_states_t[idx_f, g_index:] = goal_f

        # ---------- expert HER（按当前state最近邻抽取） ----------
        if expert_local.numel() > 0:
            idx_e = her_idx[expert_local]
            cur_state_part = states_t[idx_e, :g_index]  # shape=(n_e, g_dim)

            goal_e = self._sample_expert_goals_for_batch(cur_state_part)  # shape=(n_e, g_dim)

            # 距离过滤（可选）
            if self.config.max_expert_goal_dist > 0:
                dist = torch.norm(cur_state_part - goal_e, dim=1)
                ok = dist <= self.config.max_expert_goal_dist
                if ok.any():
                    idx_e = idx_e[ok]
                    goal_e = goal_e[ok]
                else:
                    idx_e = torch.empty((0,), dtype=torch.long)

            if idx_e.numel() > 0:
                states_t[idx_e, g_index:] = goal_e
                next_states_t[idx_e, g_index:] = goal_e

        # 重算HER样本的奖励与done
        rew_new, done_new = self._her_reward_fn(
            state=states_t[her_idx],
            action=actions_t[her_idx],
            next_state=next_states_t[her_idx],
        )

        rew_new = rew_new.to(dtype=rewards_t.dtype, device=rewards_t.device)
        done_new = done_new.to(dtype=dones_t.dtype, device=dones_t.device)

        if rew_new.ndim == 0:
            rew_new = rew_new.unsqueeze(0)
        if done_new.ndim == 0:
            done_new = done_new.unsqueeze(0)

        if rewards_t.ndim == 1:
            rewards_t[her_idx] = rew_new.reshape(-1)
        else:
            rewards_t[her_idx, 0] = rew_new.reshape(-1)

        if dones_t.ndim == 1:
            dones_t[her_idx] = done_new.reshape(-1)
            dones_t = dones_t.unsqueeze(-1)
        else:
            dones_t[her_idx, 0] = done_new.reshape(-1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    # region private methods
    def _start_a_new_trajectory(self) -> None:
        self._current_trajectory_id = self._trajectory_counter
        self._current_trajectory_start_position = self._position_to_write
        self._current_trajectory_length = 0
        self._trajectory_counter += 1

    def _write_a_transition(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor) -> None:
        p = self._position_to_write

        self._states[p].copy_(state)
        self._actions[p].copy_(action)
        self._rewards[p].copy_(reward)
        self._next_states[p].copy_(next_state)
        self._dones[p].copy_(done)

        self._trajectory_id_table[p] = self._current_trajectory_id
        self._trajectory_step_table[p] = self._current_trajectory_length
        self._current_trajectory_length += 1

        trajectory_start_position = self._current_trajectory_start_position
        trajectory_end_position = (trajectory_start_position + self._current_trajectory_length - 1) % self._capacity

        if trajectory_start_position <= trajectory_end_position:
            self._trajectory_length_table[trajectory_start_position:trajectory_end_position + 1] = self._current_trajectory_length
        else:
            self._trajectory_length_table[trajectory_start_position:] = self._current_trajectory_length
            self._trajectory_length_table[:trajectory_end_position + 1] = self._current_trajectory_length

        self._position_to_write = (self._position_to_write + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

        self._cache_dirty = True

    def _get_valid_positions(self) -> Tensor:
        if self._valid_pos_cache is not None and not self._cache_dirty:
            return self._valid_pos_cache

        if self._size == 0:
            self._valid_pos_cache = torch.empty((0,), dtype=torch.long)
            self._cache_dirty = False
            return self._valid_pos_cache

        if self._size < self._capacity:
            valid = torch.arange(0, self._size, dtype=torch.long)
        else:
            valid = (torch.arange(self._capacity, dtype=torch.long) + self._position_to_write) % self._capacity

        self._valid_pos_cache = valid
        self._cache_dirty = False
        return valid

    def _to_1d_float_tensor(self, x) -> Tensor:
        """把list/np/torch统一成CPU float32 1D Tensor"""
        if isinstance(x, Tensor):
            t = x.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x).to(dtype=torch.float32).reshape(-1)
        else:
            # list / tuple / 标量等
            t = torch.tensor(x, dtype=torch.float32).reshape(-1)
        return t

    def _rebuild_expert_cache_if_needed(self) -> None:
        if not self._expert_cache_dirty:
            return

        if self._expert_data is None or len(self._expert_data) == 0:
            self._expert_states_tensor = None
            self._expert_actions_tensor = None
            self._expert_cache_dirty = False
            return

        states_list = []
        actions_list = []

        for _, row in self._expert_data.iterrows():
            s = self._to_1d_float_tensor(row["state"])
            a = self._to_1d_float_tensor(row["action"])

            if s.numel() != self.config.state_dim:
                raise ValueError(f"expert state dim错误: {s.numel()} != {self.config.state_dim}")
            if a.numel() != self.config.action_dim:
                raise ValueError(f"expert action dim错误: {a.numel()} != {self.config.action_dim}")

            # 仅缓存state_part作为goal候选
            states_list.append(s[:self._g_dim])
            actions_list.append(a)

        self._expert_states_tensor = torch.stack(states_list, dim=0)   # (N, g_dim)
        self._expert_actions_tensor = torch.stack(actions_list, dim=0) # (N, action_dim)
        self._expert_cache_dirty = False

    def _sample_expert_goals_for_batch(self, cur_state_part: Tensor) -> Tensor:
        """
        根据当前状态批量抽取专家goal:
        - 对每个当前state, 计算到全部expert state的L2距离
        - 取最近top-k
        - 从top-k中随机选1个
        """
        assert self._expert_states_tensor is not None and self._expert_states_tensor.shape[0] > 0, "无可用专家数据"

        expert = self._expert_states_tensor  # (N, g_dim)
        n_e = cur_state_part.shape[0]
        n_expert = expert.shape[0]
        topk = max(1, min(int(self.config.expert_topk), n_expert))

        # 距离矩阵: (n_e, n_expert)
        # torch.cdist在CPU上可用，简洁且稳定
        dist = torch.cdist(cur_state_part, expert, p=2)

        # 每行取最近top-k的索引
        _, nn_idx = torch.topk(dist, k=topk, dim=1, largest=False)  # (n_e, topk)

        # 每行随机挑一个top-k邻居
        pick = torch.randint(0, topk, (n_e,), dtype=torch.long)      # (n_e,)
        row = torch.arange(n_e, dtype=torch.long)
        chosen_idx = nn_idx[row, pick]                               # (n_e,)

        goal = expert[chosen_idx]                                    # (n_e, g_dim)
        return goal
    # endregion