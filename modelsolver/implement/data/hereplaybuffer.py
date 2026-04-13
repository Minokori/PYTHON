import torch
from torch import Tensor, no_grad

from modelsolver.abc.config import ReplayBufferConfig
from modelsolver.abc.data import IReplayBuffer
from modelsolver.abc.reward import IReward
from modelsolver.implement.data.her import HEReplayConfig


class SimpleHEReplay(IReplayBuffer):
    """高性能版简单HER实现（仅扁平存储 + 可注入奖励函数）"""

    def __init__(self, config: ReplayBufferConfig, reward:IReward):
        assert issubclass(type(config), HEReplayConfig), "HER 经验回放池需要 HEReplayConfig 实例作为配置"
        # DI
        self._config = config
        self._her_reward_fn = reward


        # 容量
        self._capacity = int(self._config.capacity)
        """经验回放池的容量"""
        self._size = 0
        """当前有效样本数, 即池中所有轨迹的长度之和, 但不超过 capacity"""
        self._pos = 0
        """ring 写指针, 指向下一条写入数据的位置, 范围 [0, capacity-1]"""

        # 存储结构, 使用 Tensor 预分配, 以获得更好的性能.
        self._states = torch.empty((self.config.capacity, self.config.state_dim), dtype=torch.float32, device="cpu")
        """state存储, shape = (MAX_CAPACITY, state_dim)"""
        self._actions = torch.empty((self.config.capacity, self.config.action_dim), dtype=torch.float32, device="cpu")
        """action存储, shape = (MAX_CAPACITY, action_dim)"""
        self._rewards = torch.empty((self.config.capacity, 1), dtype=torch.float32, device="cpu")
        """reward存储, shape = (MAX_CAPACITY, 1)"""
        self._next_states = torch.empty((self.config.capacity, self.config.state_dim), dtype=torch.float32, device="cpu")
        """next_state存储, shape = (MAX_CAPACITY, state_dim)"""
        self._dones = torch.empty((self.config.capacity, 1), dtype=torch.float32, device="cpu")
        """done存储, shape = (MAX_CAPACITY, 1)"""
        # episode 信息
        self._ep_id = torch.full((self._capacity,), -1, dtype=torch.long)  # shape = (MAX_CAPACITY,)
        """轨迹ID, shape = (MAX_CAPACITY,)

        self._ep_id[i] 表示第 i 个样本所属的轨迹ID, 通过该ID可以区分不同轨迹的样本, 以便HER算法在同一轨迹内采样goal
        """
        self._ep_step = torch.zeros((self._capacity,), dtype=torch.long)      # 在该episode内的step shape  = (MAX_CAPACITY,)

        self._ep_len = torch.zeros((self._capacity,), dtype=torch.long)       # 仅对该episode有效区间正确 shape = (MAX_CAPACITY,)
        self._episode_counter = 0
        """共写入过多少条轨迹"""
        self._current_ep_id = -1
        """当前正在写入的轨迹ID, -1表示还未开始写入任何轨迹"""
        self._current_ep_start_pos = -1
        """当前正在写入的轨迹在ring中的起始位置索引, -1表示还未开始写入任何轨迹"""
        self._current_ep_len = 0
        """当前正在写入的轨迹已写入的样本数, 仅对当前轨迹有效"""

        # 可采样位置缓存（有效样本在ring中的物理索引）
        self._valid_pos_cache: Tensor | None = None
        self._cache_dirty = True

    @property
    def config(self) -> HEReplayConfig:
        return self._config  # type: ignore

    def __len__(self) -> int:
        return self._size



    def start_a_new_trajectory(self):
        """开启一条新的轨迹记录."""
        self._current_ep_id = self._episode_counter
        self._episode_counter += 1
        self._current_ep_start_pos = self._pos
        self._current_ep_len = 0

    def _write_one(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor):
        """写入一条 状态转移链

        Args:
            state (Tensor): _description_
            action (Tensor): _description_
            reward (Tensor): _description_
            next_state (Tensor): _description_
            done (Tensor): _description_
        """
        p = self._pos

        # 写入数据
        self._states[p].copy_(state)
        self._actions[p].copy_(action)
        self._rewards[p].copy_(reward)
        self._next_states[p].copy_(next_state)
        self._dones[p].copy_(done)

        # 更新轨迹元信息
        self._ep_id[p] = self._current_ep_id
        self._ep_step[p] = self._current_ep_len
        self._current_ep_len += 1

        # 更新轨迹长度信息
        ep_start = self._current_ep_start_pos
        ep_end = (ep_start + self._current_ep_len - 1) % self._capacity
        if ep_start <= ep_end: # 轨迹区间在ring中没有跨界
            self._ep_len[ep_start:ep_end + 1] = self._current_ep_len
        else:  # 轨迹区间在ring中跨界了, 需要分两段更新
            self._ep_len[ep_start:] = self._current_ep_len
            self._ep_len[:ep_end + 1] = self._current_ep_len

        self._pos = (self._pos + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

        self._cache_dirty = True

    def _get_valid_positions(self) -> Tensor:
        """获取当前池中所有有效样本在ring中的物理索引. 只有这些位置的样本才是可采样的."""
        if self._valid_pos_cache is not None and not self._cache_dirty:
            return self._valid_pos_cache

        if self._size == 0:
            self._valid_pos_cache = torch.empty((0,), dtype=torch.long)
            self._cache_dirty = False
            return self._valid_pos_cache

        if self._size < self._capacity:
            valid = torch.arange(0, self._size, dtype=torch.long)
        else:
            valid = (torch.arange(self._capacity, dtype=torch.long) + self._pos) % self._capacity

        self._valid_pos_cache = valid
        self._cache_dirty = False
        return valid

    def append(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        new: bool = False
    ) -> None:
        """向池中添加一条序列 `(s,a,r,s')` , 每个元素 shape = (1, dim)"""
        assert state.is_cpu and action.is_cpu and reward.is_cpu and next_state.is_cpu and done.is_cpu, \
            "输入的 Tensor 必须在 CPU 上"

        if new or self._current_ep_id<0:
            self.start_a_new_trajectory()
        self._write_one(state, action, reward, next_state, done)

    @property
    def can_sample(self) -> bool:
        return self._size > self._config.minimal_capacity

    @no_grad
    def sample(self, her: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = int(self.config.batch_size)
        g_index = self.config.state_dim // 2

        valid_pos = self._get_valid_positions()
        n_valid = valid_pos.numel()
        assert n_valid > 0, "buffer has no valid samples"

        # 随机采样的索引
        pos_now = valid_pos[torch.randint(0, n_valid, (batch_size,))]

        # 从池中采样一批数据
        states_t = self._states[pos_now].clone()
        actions_t = self._actions[pos_now].clone()
        rewards_t = self._rewards[pos_now].clone()
        next_states_t = self._next_states[pos_now].clone()
        dones_t = self._dones[pos_now].clone().float()

        if not her:
            if dones_t.ndim == 1:
                dones_t = dones_t.unsqueeze(-1)
            return states_t, actions_t, rewards_t, next_states_t, dones_t

        # HER 逻辑

        # her_idx: 需要进行 HER 重标注的样本在当前 batch 中的索引, shape = (her_size,)
        # her_mask = torch.rand(batch_size) < self.config.her_p
        her_idx = torch.nonzero(torch.rand(batch_size) < self.config.her_p, as_tuple=False).flatten()
        if her_idx.numel() == 0:
            if dones_t.ndim == 1:
                dones_t = dones_t.unsqueeze(-1)
            return states_t, actions_t, rewards_t, next_states_t, dones_t

        # 将要被重标注的样本在ring中的物理索引
        pos_h = pos_now[her_idx]
        step_h = self._ep_step[pos_h]
        ep_len_h = self._ep_len[pos_h]

        max_off = (ep_len_h - step_h - 1).clamp_min(0)
        u = torch.rand_like(max_off, dtype=torch.float)
        off = (u * (max_off.to(torch.float) + 1.0)).to(torch.long)

        goal_step = step_h + off
        ep_start = (pos_h - step_h) % self._capacity
        pos_goal = (ep_start + goal_step) % self._capacity

        goal_state = self._states[pos_goal]  # 目标状态, shape = (her_size, state_dim)

        # goal 重标注
        states_t_h = states_t[her_idx]
        next_states_t_h = next_states_t[her_idx]
        sampled_goal = goal_state[:, :g_index]  # 被选作 goal 的状态的state部分, shape = (her_size, state_dim/2)

        states_t_h[:, g_index:] = sampled_goal # 将state中的goal部分重标注为 sampled_goal(future.state)
        next_states_t_h[:, g_index:] = sampled_goal  # 将next_state中的goal部分重标注为 sampled_goal(future.state)

        # 保存对her_idx对应样本的重标注结果
        states_t[her_idx] = states_t_h
        next_states_t[her_idx] = next_states_t_h

        # 计算重标注后的奖励和done.
        # 输入：next_state_的state_part 与 goal_part（都为 batch）
        rew_new, done_new = self._her_reward_fn(
            state=states_t[her_idx],  # shape = (her_size, state_dim)
            action= actions_t[her_idx], # shape = (her_size, action_dim)
            next_state = next_states_t[her_idx], # shape = (her_size, state_dim/2)
            # goal= sampled_goal# shape = (her_size, state_dim/2)
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