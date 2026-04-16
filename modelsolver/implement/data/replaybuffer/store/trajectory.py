"""高效的轨迹存储池"""

import torch
from torch import Tensor


class TrajectoryStore:
    """
    仅负责:
    1) transition写入ring
    2) 轨迹元信息维护
    3) 有效位置抽样
    4) future goal位置抽样
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        """初始化按轨迹存储的状态转移链存储

        Args:
            capacity (int): 容量(能容纳多少状态转移链)
            state_dim (int): 状态的特征维度(包含GOAL.)
            action_dim (int): 动作维度
        """
        self._capacity = int(capacity)
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)

        # transition buffers
        self._states = torch.empty((self._capacity, self._state_dim), dtype=torch.float32, device="cpu")
        self._actions = torch.empty((self._capacity, self._action_dim), dtype=torch.float32, device="cpu")
        self._rewards = torch.empty((self._capacity, 1), dtype=torch.float32, device="cpu")
        self._next_states = torch.empty((self._capacity, self._state_dim), dtype=torch.float32, device="cpu")
        self._dones = torch.empty((self._capacity, 1), dtype=torch.float32, device="cpu")

        # trajectory metadata
        self._trajectory_id_table = torch.full((self._capacity,), -1, dtype=torch.long)
        self._trajectory_step_table = torch.zeros((self._capacity,), dtype=torch.long)
        self._trajectory_length_table = torch.zeros((self._capacity,), dtype=torch.long)

        # ring pointers
        self._size = 0
        self._position_to_write = 0

        # current trajectory context
        self._trajectory_counter = 0
        self._current_trajectory_id = -1
        self._current_trajectory_start_position = -1
        self._current_trajectory_length = 0

        # cache
        self._valid_pos_cache: Tensor|None = None
        self._cache_dirty = True


    def __getitem__(self, index:int|slice|Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self._states[index].clone().reshape(-1,self._state_dim),
            self._actions[index].clone().reshape(-1,self._action_dim),
            self._rewards[index].clone().reshape(-1,1),
            self._next_states[index].clone().reshape(-1,self._state_dim),
            self._dones[index].clone().reshape(-1,1),
        )


    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def start_new_trajectory(self) -> None:
        """开启一条新轨迹.

        *这会将当前轨迹上下文切换到新轨迹, 但不会修改ring buffer中的数据.
        注意: 这不会强制要求下一条写入的数据必须是新轨迹的开始, 也不会强制要求新轨迹必须从环境重置开始.
        这是一个纯粹的上下文切换操作, 由调用者根据实际情况决定何时调用.*"""
        self._current_trajectory_id = self._trajectory_counter
        self._current_trajectory_start_position = self._position_to_write
        self._current_trajectory_length = 0
        self._trajectory_counter += 1

    def append_transition(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> None:
        """写入一条状态转移链.

        Args:
            state (Tensor): 状态, shape = (state_dim,) **注意: state应包含goal信息, 即 state_dim = obs_dim + goal_dim**
            action (Tensor): 动作, shape = (action_dim,)
            reward (Tensor): 奖励, shape = (1,)
            next_state (Tensor): 下一状态, shape = (state_dim,)
            done (Tensor): 是否结束, shape = (1,)

            ---

            new (bool): 是否为新轨迹的开始. 默认为 False, 即默认添加到当前轨迹中. 设置为 True 时, 将在池中添加一条新轨迹.
        """
        if self._current_trajectory_id<0:
            self.start_new_trajectory()


        p = self._position_to_write
        self._states[p].copy_(state.reshape(-1))
        self._actions[p].copy_(action.reshape(-1))
        self._rewards[p].copy_(reward.reshape(-1))
        self._next_states[p].copy_(next_state.reshape(-1))
        self._dones[p].copy_(done.reshape(-1))

        # 更新轨迹信息
        self._trajectory_id_table[p] = self._current_trajectory_id
        self._trajectory_step_table[p] = self._current_trajectory_length
        self._current_trajectory_length += 1

        trajectory_start = self._current_trajectory_start_position
        trajectory_end = (trajectory_start + self._current_trajectory_length - 1) % self._capacity

        if trajectory_start <= trajectory_end:
            self._trajectory_length_table[trajectory_start:trajectory_end + 1] = self._current_trajectory_length
        else:
            self._trajectory_length_table[trajectory_start:] = self._current_trajectory_length
            self._trajectory_length_table[:trajectory_end + 1] = self._current_trajectory_length

        # ring前进
        self._position_to_write = (self._position_to_write + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

        self._cache_dirty = True





    def sample_future_goal_positions(self, positions: Tensor) -> Tensor:
        """给定一批物理位置, 从同轨迹未来(含当前)随机采样goal位置.

        Args:
            positions (Tensor): 状态转移链索引. shape = (B,)

        Returns:
            goal_position (Tensor): 作为 goal 的未来状态的索引. shape = (B,)
        """
        step = self._trajectory_step_table[positions]  # shape: (B,)
        ep_len = self._trajectory_length_table[positions]  # shape: (B,)

        max_off = (ep_len - step - 1).clamp_min(0) # shape: (B,)
        u = torch.rand_like(max_off, dtype=torch.float32)
        off = (u * (max_off.to(torch.float32) + 1.0)).to(torch.long)

        goal_step = step + off
        ep_start = (positions - step) % self._capacity
        goal_pos = (ep_start + goal_step) % self._capacity
        return goal_pos

    def get_states_by_positions(self, positions: Tensor) -> Tensor:
        return self._states[positions]

    #region private methods
    def get_valid_positions(self) -> Tensor:
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
    #endregion
