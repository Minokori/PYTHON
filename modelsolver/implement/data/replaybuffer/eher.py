"""融合HER和专家重标注的经验回放池"""

import logging
from dataclasses import dataclass

import pandas as pd
import torch
from dataclasses_json import dataclass_json
from torch import Tensor, no_grad

from modelsolver.abc.config import ReplayBufferConfig
from modelsolver.abc.data import IReplayBuffer
from modelsolver.abc.reward import IReward
from modelsolver.implement.data.replaybuffer.hereplaybuffer import HERConfig
from modelsolver.implement.data.replaybuffer.store.expert import ExpertStore
from modelsolver.implement.data.replaybuffer.store.trajectory import \
    TrajectoryStore


logging.basicConfig(level=logging.INFO, filename="./eher.log", filemode="w", encoding="utf-8")
@dataclass_json
@dataclass
class EHERConfig(HERConfig):
    """HER经验回放池配置"""
    expert_data_path: str = ""
    """pkl 格式的专家数据路径, 数据应包含'state'和'action'两列"""
    expert_p: float = 0.5
    """在HER命中的样本中, 使用专家goal的概率"""
    expert_topk: int = 32
    """按当前state找最近邻后, 在top-k中随机采样"""
    max_expert_goal_dist: float = -1.0
    """专家goal距离过滤阈值(L2). <0表示不启用"""
    expert_state_col: str = "state"
    """专家数据中状态列的名称"""
    expert_action_col: str = "action"
    """专家数据中动作列的名称"""

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
        open("./eher.log", "w", encoding="utf-8").close()  # 每次初始化时清空日志文件
        self._config = config
        self._her_reward_fn = reward

        self._g_dim = int(self._config.state_dim // 2)

        self._store = TrajectoryStore(
            capacity=int(self._config.capacity),
            state_dim=int(self._config.state_dim),
            action_dim=int(self._config.action_dim),
        )

        self._expert_store = ExpertStore(
            state_info=(self.config.expert_state_col, self.config.state_dim),
            action_info=(self.config.expert_action_col, self.config.action_dim),
            df=pd.read_pickle(self.config.expert_data_path))


        # 专家数据计算奖励.
        with torch.no_grad():
            expert_states = self._expert_store.expert_state  # shape = (N, state_dim)
            expert_rewards, _ = self._her_reward_fn(next_state = expert_states)
            self.expert_rewards = expert_rewards

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
        # TODO 代码不报错, 需要检查业务逻辑是否有问题
        # 变量名称: 带 pos 的表示实际物理索引, 带 idx 的表示其他索引.
        # 基础随机抽样
        pos_now = self._store.sample_postions(self.config.batch_size)  # shape = (batch_size,)
        states_t, actions_t, rewards_t, next_states_t, dones_t = self._store[pos_now]

        # HER索引抽取. her_idx: shape = (her),dtype=int 表示一个batch中哪些会被HER.
        her_idx = torch.arange(self.config.batch_size)[torch.rand(self.config.batch_size) < self.config.her_p]

        # 不采用HER或不满足HER条件时, 直接返回原始采样结果
        if her_idx.numel() == 0 or not her:
            return states_t, actions_t, rewards_t, next_states_t, dones_t

        # HER idx切分: e_idx用于专家重标注, f_idx用于future重标注.
        # 例如: her_idx = [0,2,5,7], expert_p=0.5时, 可能切分结果为 e_idx=[2,7], f_idx=[0,5]
        e_idx, f_idx = self._split_her_idx(her_idx)

        # 从future抽取HER样本
        future_pos = self._store.sample_future_goal_positions(pos_now[f_idx])
        future_ob = self._store.get_states_by_positions(future_pos)[:, :self.config.goal_index]
        states_t[f_idx, self.config.goal_index:] = future_ob
        next_states_t[f_idx, self.config.goal_index:] = future_ob

        # 从专家数据抽取HER样本
        expert_state, pos_expert = self._expert_store.sample_expert_states(states_t[e_idx],self.config.expert_topk)

        #  阈值过滤: 仅保留距离当前状态较近的专家goal
        ok = self._select_valid_expert_goals(states_t[e_idx], expert_state, pos_expert)
        e_idx = e_idx[ok]

        if e_idx.numel() > 0:
            states_t[e_idx, self.config.goal_index:] = expert_state[:,:self.config.goal_index]
            next_states_t[e_idx, self.config.goal_index:] = expert_state[:,:self.config.goal_index]

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
        """将HER索引切分为两部分, 一部分用于从future抽取, 一部分用于在专家数据中抽取

        Args:
            her_idx (Tensor): 表示一个batch中索引为多少的数据会被HER重标注, shape = (her,) dtype=int

        Returns:
            expert_her_idx&future_her_idx (tuple[Tensor, Tensor]): 专家重标注索引和future重标注索引, shape = (expert_her_batch,), (future_her_batch,). expert_her_batch + future_her_batch = her_batch.
        """
        mask_expert = torch.rand(her_idx.numel()) < self.config.expert_p  # shape = (her_batch,) dtype=bool
        expert_local = torch.nonzero(mask_expert, as_tuple=False).flatten()
        future_local = torch.nonzero(~mask_expert, as_tuple=False).flatten()
        return her_idx[expert_local], her_idx[future_local]

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

        ob_reward,_ = self._her_reward_fn(next_state=goals_ob)
        e_reward = self.expert_rewards[chosen_indices]

        ok2 = (ob_reward <= e_reward + 1e-3).reshape(-1)  # 仅当专家reward更优时才使用专家goal

        ok_final = ok & ok2

        # ok : 合适的专家goal布尔索引 (在 goals_e 里的索引), shape = (num_expert_goals,)
        return ok_final