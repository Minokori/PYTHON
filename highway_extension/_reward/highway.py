"""高速公路适用的奖励函数"""
# region imports
import logging

from torch import Tensor, minimum, no_grad

from modelsolver.abc.reward import IReward


#endregion
logging.basicConfig(level=logging.INFO, filename="./logs/reward.log", filemode="w", encoding="utf-8")

class HighwayReward(IReward):
    """连续高速公路场景的奖励函数.

    由 和*周边车辆*的状态差异 和 *自车*状态相较于*任务目标*的差异 两部分相加得到.
    """

    @property
    def is_learnable(self) -> bool:
        return False
    @no_grad
    def forward(self, **kwargs: Tensor) -> tuple[Tensor, Tensor]:
        """
        计算奖励函数, 返回奖励和终止标志

        Args:
            kwargs (dict): 包含 state 或 next_state 的字典, shape = (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]


        Returns:
            rewards (Tensor): 奖励(越高越好, 正值), shape = (B,1)
            done (Tensor): 终止标志, 1表示成功, 0表示未终止, -1表示失败, shape = (B,1)

        """
        if "expert" in kwargs:
            raise NotImplementedError("为 EHER预留的接口, 还没有实现相关逻辑.")
        else:
            state = kwargs["state"]
            obs_ego = state[0:7] # shape = (B, 7) / (1, 7)
            obs_around = state[7:112] # shape = (B, 105) / (1, 105)
            goal = state[112:] # shape = (B, 7) / (1, 7)
            # 计算奖励函数
            lane_keep_reward = self._lane_keep_reward(obs_ego)


        raise NotImplementedError("还没有实现奖励函数的逻辑, 需要根据任务目标和自车状态来计算奖励.")



    def _lane_keep_reward(self, obs_ego: Tensor) -> Tensor:
        """
        车道保持: 对横向偏移做稠密惩罚, 让车保持在车道中心附近.


        *左偏移和右偏移不对等, 倾向于让车向右换道*
        Args:
            obs_ego (Tensor): 自车状态, shape = (B, 7)

        Returns:
            reward (Tensor): 奖励(越高越好, 正值), shape = (B,1)
        """
        y_position = obs_ego[:, 2]*12 # 真实世界坐标, shape = (B,1)

        lane_index = ((y_position-3.7/2)/3.7).int() # 车道索引, shape = (B,1) 车道宽 = 3.7m, 车道中心线y坐标 = 3.7/2 + lane_index*3.7

        raise NotImplementedError("还没有实现车道保持奖励函数的逻辑, 需要根据车道索引和横向偏移来计算奖励.")
