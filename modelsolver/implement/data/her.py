"""Hindsight Experience Replay (HER) 实现"""


from modelsolver.abc.data import IReplayBuffer
import numpy as np
from torch import Tensor


class SimpleHEReplay(IReplayBuffer):
    """简单的HER实现 (没有使用环境的reward函数，而是直接计算HER奖励)"""

    def __init__(self):

        pass


    def sample(self, **kwargs) -> tuple[Tensor,...]:

        her = kwargs.get("her", False)
        if her:
            k = kwargs.get("k", None)
            if k is None:
                raise ValueError("HER采样方法需要指定k参数")

        match her:
            case False:
                return super().sample()
            case True:
                return self._her_sample(k)
            case _:
                raise ValueError("HER采样方法的her参数必须为True或False")

    def _her_sample(self, k: int) -> tuple[Tensor,...]:
        state, action, reward, next_state, done = super().sample()

        her_state, her_action, her_reward, her_next_state,her_done, her_goal = [], [], [], [], [], []

        for i in range(self.config.batch_size):
            future_idx = np.random.randint(i, len(self), size = k)
            for idx in future_idx:
                new_goal = self._next_state_buffer[idx]

                her_state.append(state[i])
                her_action.append(action[i])
                her_reward.append(-np.linalg.norm(next_state[i]-new_goal))  # 计算HER奖励
                her_next_state.append(next_state[i])
                her_done.append(done[i])
                her_goal.append(new_goal)

        return Tensor(her_state), Tensor(her_action), Tensor(her_reward), Tensor(her_next_state), Tensor(her_done), Tensor(her_goal)
