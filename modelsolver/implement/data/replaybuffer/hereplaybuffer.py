from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json
from torch import Tensor, no_grad

from modelsolver.abc.config import ReplayBufferConfig
from modelsolver.abc.data import IReplayBuffer
from modelsolver.abc.reward import IReward
from modelsolver.implement.data.replaybuffer.store.trajectory import \
    TrajectoryStore


@dataclass_json
@dataclass
class HERConfig(ReplayBufferConfig):
    """HER经验回放池配置"""
    her_p: float = 0.8  # HER采样的概率
    """HER采样的概率, 即在采样时, 以 her_p 的概率使用HER算法生成新的样本, 否则使用原始样本"""

    @property
    def goal_index(self) -> int:
        """状态中goal信息的起始索引, (state_dim // 2)"""
        return self.state_dim // 2


class HEReplayBuffer(IReplayBuffer):
    """简单HER经验回放池实现"""

    def __init__(self, config: ReplayBufferConfig, reward:IReward):
        assert issubclass(type(config), HERConfig), "HER 经验回放池需要 HEReplayBufferConfig 实例作为配置"
        # DI
        self._config = config
        self._her_reward_fn = reward
        self._store = TrajectoryStore(self._config.capacity, self._config.state_dim, self._config.action_dim)


        # 容量
        self._capacity = int(self._config.capacity)
        """经验回放池的容量"""
        self._size = 0
        """当前有效样本数, 即池中所有轨迹的长度之和, 但不超过 capacity"""
        self._position_to_write = 0
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
        self._trajectory_id_table = torch.full((self._capacity,), -1, dtype=torch.long)  # shape = (MAX_CAPACITY,)
        """shape = (MAX_CAPACITY,)

        `[i]` 表示第 i 个样本所属的轨迹ID, 通过该ID可以区分不同轨迹的样本, 以便HER算法在同一轨迹内采样goal
        """
        self._trajectory_step_table = torch.zeros((self._capacity,), dtype=torch.long)      # 在该episode内的step shape  = (MAX_CAPACITY,)
        """shape = (MAX_CAPACITY,)

        `[i]` 表示第 i 个样本**在其所属轨迹中的**step, 从0开始计数. 通过该step信息可以知道一个样本在轨迹中的位置, 以便HER算法采样未来的goal
        """

        self._teajectory_length_table = torch.zeros((self._capacity,), dtype=torch.long)       # 仅对该episode有效区间正确 shape = (MAX_CAPACITY,)
        """shape = (MAX_CAPACITY,)

        `[i]` 表示第 i 个样本所属轨迹的总长度. 通过该信息可以知道一个样本所在轨迹的长度, 以便HER算法采样未来的goal"""
        self._trajectory_counter = 0
        """共写入过多少条轨迹"""
        self._current_trajectory_id = -1
        """当前正在写入的轨迹ID, -1表示还未开始写入任何轨迹"""
        self._current_trajectory_start_position = -1
        """当前正在写入的轨迹在ring中的起始位置索引, -1表示还未开始写入任何轨迹"""
        self._current_trajectory_length = 0
        """当前正在写入的轨迹已写入的样本数, 仅对当前轨迹有效"""

        # 可采样位置缓存（有效样本在ring中的物理索引）
        self._valid_pos_cache: Tensor | None = None
        self._cache_dirty = True
    @property
    def can_sample(self) -> bool:
        return self._store.size > self._config.minimal_capacity

    @property
    def config(self) -> HERConfig:
        return self._config  # type: ignore

    def __len__(self) -> int:
        return self._store.size


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

        if new:
            self._store.start_new_trajectory()
        self._store.append_transition(state, action, reward, next_state, done)

    @no_grad
    def sample(self, her: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # 基础随机抽样
        pos_now = self._store.sample_postions(self.config.batch_size)
        states_t, actions_t, rewards_t, next_states_t, dones_t = self._store[pos_now]


        # HER索引抽取
        her_idx = torch.nonzero(torch.rand(self.config.batch_size) < self.config.her_p, as_tuple=False).flatten()  # shape = (her, )

        # 不采用HER或不满足HER条件时, 直接返回原始采样结果
        if her_idx.numel() == 0 or not her:
            return states_t, actions_t, rewards_t, next_states_t, dones_t


        # her_idx: 需要进行 HER 重标注的样本在当前 batch 中的索引, shape = (her_size,)
        goal_pos = self._store.sample_future_goal_positions(pos_now[her_idx])
        goal_f = self._store.get_states_by_positions(goal_pos)[:, :self.config.goal_index]

        # future relabel
        states_t[her_idx, self.config.goal_index:] = goal_f
        next_states_t[her_idx, self.config.goal_index:] = goal_f

        # 计算重标注后的奖励和done.
        # 输入：next_state_的state_part 与 goal_part（都为 batch）
        rew_new, done_new = self._her_reward_fn(
            state=states_t[her_idx],  # shape = (her_size, state_dim)
            action= actions_t[her_idx], # shape = (her_size, action_dim)
            next_state = next_states_t[her_idx], # shape = (her_size, state_dim/2)
            # goal= sampled_goal# shape = (her_size, state_dim/2)
             )
        rewards_t[her_idx] = rew_new
        dones_t[her_idx] = done_new

        return states_t, actions_t, rewards_t, next_states_t, dones_t