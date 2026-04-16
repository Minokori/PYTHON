from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json
from torch import Tensor, no_grad

from modelsolver.abc.config import ReplayBufferConfig
from modelsolver.abc.data import IReplayBuffer
from modelsolver.abc.reward import IReward


@dataclass_json
@dataclass
class HEReplayBufferConfig(ReplayBufferConfig):
    """HER经验回放池配置。"""

    her_p: float = 0.8
    """HER采样概率。"""

    her_k_future: int = 20
    """future窗口大小（仅看后k步）。"""

    her_exclude_failture: bool = True
    """是否禁止将 done 状态作为 goal。"""

    her_future_strategy: str = "uniform"
    """future采样策略：uniform | exp。"""


class SimpleHEReplay(IReplayBuffer):
    """高性能版简单HER实现（扁平存储 + 可注入奖励函数）。"""




    def __init__(self, config: ReplayBufferConfig, reward: IReward):
        assert issubclass(type(config), HEReplayBufferConfig), "HER 经验回放池需要 HEReplayBufferConfig 实例作为配置"

        # region DI注入配置和HER奖励函数
        self._config = config
        self._her_reward_fn = reward
        # endregion

        # region 内部状态初始化, 容量, size, 写指针.
        self._capacity = int(self._config.capacity)
        """经验回放池的容量"""
        self._size = 0
        """当前有效样本数, 即池中所有轨迹的长度之和, 但不超过 capacity"""
        self._position_to_write = 0
        """ring 写指针, 指向下一条写入数据的位置, 范围 [0, capacity-1]"""
        # endregion

        # region 初始化存储结构, 使用 Tensor 预分配, 提高 sample 性能
        self._states = torch.empty((self.config.capacity, self.config.state_dim), dtype=torch.float32, device="cpu")
        """state 存储, shape=(capacity, state_dim)"""
        self._actions = torch.empty((self.config.capacity, self.config.action_dim), dtype=torch.float32, device="cpu")
        """action 存储, shape=(capacity, action_dim)"""
        self._rewards = torch.empty((self.config.capacity, 1), dtype=torch.float32, device="cpu")
        """reward 存储, shape=(capacity, 1)"""
        self._next_states = torch.empty((self.config.capacity, self.config.state_dim), dtype=torch.float32, device="cpu")
        """next_state 存储, shape=(capacity, state_dim)"""
        self._dones = torch.empty((self.config.capacity, 1), dtype=torch.float32, device="cpu")
        """done 标记存储, shape=(capacity, 1)"""
        # endregion

        # region 初始化轨迹存储辅助信息（用于HER在同一轨迹中抽future goal）
        self._trajectory_id_table = torch.full((self._capacity,), -1, dtype=torch.long)
        """每条 transition 对应的轨迹ID, shape=(capacity,)"""
        self._trajectory_step_table = torch.zeros((self._capacity,), dtype=torch.long)
        """每条 transition 在其轨迹中的step数, shape=(capacity,)"""
        self._trajectory_length_table = torch.zeros((self._capacity,), dtype=torch.long)
        """每条 transition 所属轨迹的长度, shape=(capacity,)"""

        self._trajectory_counter = 0
        """共写入过多少条轨迹"""
        self._current_trajectory_id = -1
        """当前正在写入的轨迹ID, *-1表示还未开始写入任何轨迹*"""
        self._current_trajectory_start_position = -1
        """当前正在写入的轨迹在存储向量中的起始位置索引, *-1表示还未开始写入任何轨迹*"""
        self._current_trajectory_length = 0
        """当前正在写入的轨迹已写入的样本数, 仅对当前轨迹有效"""

        # 可采样物理位置缓存
        self._valid_pos_cache: Tensor | None = None
        self._cache_dirty = True
        # endregion

    @property
    def config(self) -> HEReplayBufferConfig:
        """返回类型收窄后的配置对象。"""
        return self._config  # type: ignore

    def __len__(self) -> int:
        """返回当前有效样本数。"""
        return self._size

    def __getitem__(self, index: int|slice) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """获取 trasition

        Args:
            index (_Index | tuple[_Index, ...]): Tensor 索引

        Returns:
            转移链 (tuple[Tensor, Tensor, Tensor, Tensor, Tensor]): (s, a, r, s', done)
        """
        s = self._states.__getitem__(index)
        a = self._actions.__getitem__(index)
        r = self._rewards.__getitem__(index)
        next_s = self._next_states.__getitem__(index)
        d = self._dones.__getitem__(index)

        return s.clone(), a.clone(), r.clone(), next_s.clone(), d.clone()

    def append(self,state: Tensor,action: Tensor,reward: Tensor,next_state: Tensor,done: Tensor,new: bool = False
    ) -> None:
        """
        追加一条 transition 到回放池。

        Args:
            state: 当前状态，shape=(state_dim,)
            action: 动作，shape=(action_dim,)
            reward: 奖励，shape=(1,)
            next_state: 下一状态，shape=(state_dim,)
            done: 终止标记，shape=(1,)
            new: 是否开启新轨迹
        """
        assert state.is_cpu and action.is_cpu and reward.is_cpu and next_state.is_cpu and done.is_cpu, "输入的 Tensor 必须在 CPU 上"

        if new: self._start_new_trajectory()

        self._write_transition(state, action, reward, next_state, done)

    @property
    def can_sample(self) -> bool:
        """是否达到最小采样容量。"""
        return self._size > self._config.minimal_capacity

    @no_grad
    def sample(self, her: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        从回放池采样一个batch；可选HER重标注。

        Args:
            her: 是否开启HER重标注。

        Returns:
            (states, actions, rewards, next_states, dones)
        """
        # 参数
        batch_size = int(self.config.batch_size)
        goal_start_index = self.config.state_dim // 2


        # 随机采样batch个状态转移链, 作为当前 Trans
        sampled_positions = self._sample_physical_positions(batch_size)
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = self[sampled_positions]

        # 不使用 HER 重标注, 直接返回采样结果
        if not her:
            normalized_done_tensor = self._ensure_done_2d(sampled_dones)
            return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, normalized_done_tensor


        # 标记出采样出的batch内哪些样本会被 HER重标注
        her_batch_indices = self._sample_her_batch_indices(batch_size)

        # 如果没有样本被选中进行 HER 重标注, 则直接返回采样结果
        if her_batch_indices.numel() == 0:
            normalized_done_tensor = self._ensure_done_2d(sampled_dones)
            return sampled_states, sampled_actions, sampled_rewards, sampled_next_states, normalized_done_tensor


        # 为需要 HER 重标注的样本抽取 future goal 对应的位置索引
        goal_positions = self._sample_goal_positions_for_her(sampled_positions, her_batch_indices)

        # 取得作为 goal 的目标转移链信息（后续会用来重标注 state/next_state 以及重算 reward/done）
        goal_states, goal_actions, goal_rewards, goal_next_states, goal_dones = self[goal_positions]



        # 基于抽取到的 goal 位置, 构建重标注后的 state/next_state
        sampled_states[her_batch_indices, 0:goal_start_index] = goal_states[:, :goal_start_index]
        # sampled_ action 保持不变
        sampled_next_states[her_batch_indices, 0:goal_start_index] = goal_states[:, :goal_start_index]
        generated_rewards, generated_dones = self._her_reward_fn(
            state=sampled_states[her_batch_indices],
            action=sampled_actions[her_batch_indices],
            next_state=sampled_next_states[her_batch_indices],
            done = goal_dones  # 传入goal的done作为辅助信息，用于指示这个goal.state 是否是一个不好的 state
        )
        sampled_rewards[her_batch_indices,0:1] = generated_rewards
        sampled_dones[her_batch_indices,0:1] = generated_dones

        # 将 done 保证为二维列向量后返回
        normalized_done_tensor = self._ensure_done_2d(updated_dones)
        return relabeled_states, sampled_actions, updated_rewards, relabeled_next_states, normalized_done_tensor

    # region private methods

    def _start_new_trajectory(self) -> None:
        """
        开启一条新轨迹并重置“当前轨迹”状态。
        """
        self._current_trajectory_id = self._trajectory_counter
        self._current_trajectory_start_position = self._position_to_write
        self._current_trajectory_length = 0
        self._trajectory_counter += 1

    def _write_transition(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor, done: Tensor) -> None:
        """
        写入单条 transition，并更新轨迹表与ring指针。

        说明：
        - 该方法属于“存储写入”路径，按设计会修改内部缓存（这是必要副作用）。
        - 轨迹长度表会回填当前轨迹区间的长度，供HER future采样使用。
        """
        current_write_position = self._position_to_write

        # region 写入到指定位置
        self._states[current_write_position].copy_(state)
        self._actions[current_write_position].copy_(action)
        self._rewards[current_write_position].copy_(reward)
        self._next_states[current_write_position].copy_(next_state)
        self._dones[current_write_position].copy_(done)
        # endregion

        # 更新轨迹表
        self._trajectory_id_table[current_write_position] = self._current_trajectory_id
        self._trajectory_step_table[current_write_position] = self._current_trajectory_length

        # 更新当前正在写入轨迹已写入的样本数
        self._current_trajectory_length += 1

        # 更新轨迹长度表：当前轨迹的起始位置 ~ 当前写入位置（闭区间）都更新为当前轨迹长度
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

    def _sample_physical_positions(self, batch_size: int) -> Tensor:
        """从有效样本位置中随机抽取 batch 对应的物理索引

        Args:
            batch_size (int): 批量大小

        Returns:
            物理索引 (Tensor): shape=(batch_size,) 的ring物理位置
        """
        valid_positions = self._get_valid_positions()
        number_of_valid_positions = valid_positions.numel()
        assert number_of_valid_positions > 0, "buffer has no valid samples"

        random_valid_indices = torch.randint(0, number_of_valid_positions, (batch_size,))
        return valid_positions[random_valid_indices]

    def _sample_her_batch_indices(self, batch_size: int) -> Tensor:
        """
        生成当前batch内需要做HER重标注的样本下标

        Args:
            batch_size (int): 批量大小

        Returns:
            样本下标 (Tensor): shape=(her_size,) 的 batch 内索引
        """
        her_mask = torch.rand(batch_size) < self.config.her_p
        return torch.nonzero(her_mask, as_tuple=False).flatten()

    def _sample_goal_positions_for_her(self, sampled_positions: Tensor, her_batch_indices: Tensor) -> Tensor:
        """
        为HER样本抽取 future goal 对应的位置索引。

        关键逻辑：
        1) 仅在同轨迹中，从当前step之后采样；
        2) 偏移量范围受 her_k_future 限制；
        3) 可按 her_exclude_failture 过滤 failure；
        4) 按策略（uniform/exp）加权抽样。


        Args:
            sampled_positions (Tensor): 当前batch采样的物理位置, shape=(batch_size,)
            her_batch_indices (Tensor): 需要HER重标注的样本在batch内的索引, shape=(her_size,)

        Returns:
            shape=(her_size,) 的 goal 物理位置。
        """
        her_physical_positions = sampled_positions[her_batch_indices]

        # 查表得到每个样本在其轨迹中的 step，以及该轨迹总长度
        trajectory_steps = self._trajectory_step_table[her_physical_positions]
        trajectory_lengths = self._trajectory_length_table[her_physical_positions]
        max_future_window = self.config.her_k_future

        # 每个样本“理论可往后走”的最大步数 = ep_len - step - 1（至少为0）
        max_offsets_per_sample = (trajectory_lengths - trajectory_steps - 1).clamp_min(0)
        # 得到实际可采样 future 上限（受 her_k_future 限制）
        max_offsets_per_sample = torch.minimum(max_offsets_per_sample, torch.full_like(max_offsets_per_sample, max_future_window))

        # 候选偏移
        candidate_offsets = torch.arange(1, int(max_future_window) + 1).unsqueeze(0)
        # 有效便宜量掩码，shape=(her_size, K)，表示每个样本哪些偏移是合法的
        valid_offset_mask = candidate_offsets <= max_offsets_per_sample.unsqueeze(1)

        # 轨迹起点
        trajectory_start_positions = (her_physical_positions - trajectory_steps) % self._capacity

        # 最终对应的 goal 位置
        candidate_positions = (trajectory_start_positions.unsqueeze(1) + (trajectory_steps.unsqueeze(1) + candidate_offsets)) % self._capacity

        # 过滤失败的goal
        if self.config.her_exclude_failture:
            # 原逻辑：仅过滤 done<0（failure）
            candidate_done_values = self._dones[candidate_positions].squeeze(-1)
            failure_mask = candidate_done_values < 0
            valid_offset_mask = valid_offset_mask & (~failure_mask)

        # 给不同的候选偏移量按策略加权
        offset_weights = self._build_offset_weights(
            valid_offset_mask=valid_offset_mask,
            candidate_offsets=candidate_offsets,
            candidate_positions=candidate_positions
        )
        normalized_probs = self._normalize_weights(offset_weights)

        # 按权重采样偏移量, 得到最终的 goal 位置
        sampled_offsets = torch.multinomial(normalized_probs, num_samples=1).squeeze(1)
        goal_steps = trajectory_steps + sampled_offsets
        goal_positions = (trajectory_start_positions + goal_steps) % self._capacity
        return goal_positions

    def _build_offset_weights(self, valid_offset_mask: Tensor, candidate_offsets: Tensor, candidate_positions: Tensor) -> Tensor:
        """
        按配置生成future offset的采样权重。

        Returns:
            shape=(her_size, K) 的非负权重矩阵。
        """
        if self.config.her_future_strategy == "uniform":
            return valid_offset_mask.float()

        if self.config.her_future_strategy == "exp":
            # 越近权重越高（保留原业务逻辑）
            decay_factor = 0.9
            base_decay_weights = decay_factor ** candidate_offsets.float()

            # 保留原逻辑中的 success bonus
            candidate_done_values = self._dones[candidate_positions].squeeze(-1)
            success_bonus_mask = (candidate_done_values > 0).float()
            success_weight_scale = 2.0

            return base_decay_weights * (1.0 + success_weight_scale * success_bonus_mask) * valid_offset_mask.float()

        raise ValueError("unknown her_future_strategy")

    @staticmethod
    def _normalize_weights(weights: Tensor) -> Tensor:
        """
        将权重归一化为概率；若某行全0，按原逻辑将分母置1避免NaN。

        Returns:
            与输入同shape的概率张量。
        """
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights_sum[weights_sum == 0] = 1.0
        return weights / weights_sum

    @staticmethod
    def _ensure_done_2d(done_tensor: Tensor) -> Tensor:
        """
        保证done为二维列向量。

        Returns:
            shape=(B,1) 的done张量。
        """
        if done_tensor.ndim == 1:
            return done_tensor.unsqueeze(-1)
        return done_tensor

    def _get_valid_positions(self) -> Tensor:
        """
        返回当前所有有效样本在ring中的物理位置。

        说明：
        - size < capacity: 有效区间是 [0, size)
        - size == capacity: 有效区间从写指针开始绕环
        """
        if self._valid_pos_cache is not None and not self._cache_dirty:
            return self._valid_pos_cache

        if self._size == 0:
            self._valid_pos_cache = torch.empty((0,), dtype=torch.long)
            self._cache_dirty = False
            return self._valid_pos_cache

        if self._size < self._capacity:
            valid_positions = torch.arange(0, self._size, dtype=torch.long)
        else:
            valid_positions = (torch.arange(self._capacity, dtype=torch.long) + self._position_to_write) % self._capacity

        self._valid_pos_cache = valid_positions
        self._cache_dirty = False
        return valid_positions

    # endregion