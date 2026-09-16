"""高速公路适用的奖励函数"""
# region imports
import logging

import torch
from torch import Tensor, no_grad

from modelsolver.abc.reward import IReward


#endregion
logging.basicConfig(level=logging.INFO, filename="./logs/reward.log", filemode="w", encoding="utf-8")

class HighwayReward(IReward):
    """连续高速公路场景的奖励函数.

    由 和*周边车辆*的状态差异 和 *自车*状态相较于*任务目标*的差异 两部分相加得到.
    """
    # region constants
    _B = Tensor([0,0, 9000, 3066.57-3.70, 0, 0, 0])
    """
    标准化输入时用的偏移量

    [速度,航向角, x, y,  加速度x, 加速度y, lane] 的最小值
    """
    _W = Tensor([1/(120/3.6), 1.0, 1/2000, 1/(3.7*3),  1/3.6, 1/3.6, 1/2])
    """
    标准化输入时用的缩放因子

    [速度, x, y, 航向角, 加速度x, 加速度y, lane] 的单位修正权重
    """

    WEIGHT_AROUND = Tensor([0.5, 0.0, 1.0, 1.0, 0.05, 0.05, 1.0])
    """
    计算奖励的周车部分时, 各维度的权重, 用于计算周车的价值

    [速度, 航向角, x, y, 加速度x, 加速度y, lane] 的权重
    """
    WEIGHT_EGO = Tensor([0.5, 0.05, 1.0, 0.0, 0.05, 0.05, 1.0])
    """
    计算奖励的自车部分时, 各维度的权重, 用于计算自车的价值

    [速度, 航向角, x, y, 加速度x, 加速度y, lane] 的权重
    """
    GOAL = Tensor([[1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]]) # (1,7)
    """
    目标状态(**仅包含自车, 已经标准化**)

    [速度, 航向角, x, y, 加速度x, 加速度y, lane]
    """
    # endregion

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

        # 计算专家状态的奖励
        if "expert" in kwargs:  # 专家数据, shape = (B, 112), 且没有标准化
            x = self._standardize(kwargs["expert"].reshape(-1, 16, 7))  # shape = (B, 16, 7)
            return self._reward_by_obs(x)

        # 计算实际状态的奖励
        if "state"  in kwargs:  # 实际数据, shape = (B, 224), 已经标准化
            return self._reward_by_obs(kwargs["state"][:,0:112].reshape(-1, 16, 7))
        raise ValueError("必须提供合法的kwargs参数")

    @no_grad
    def _standardize(self, observations:Tensor)->Tensor:
        """对观测值进行标准化处理

        (B, 16, 7) -> (B, 16, 7)

        Args:
            observations (np.ndarray): shape =  (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]

        Returns:
            standardized_observations (np.ndarray): shape =  (B, 16, 7), 标准化后的观测值
        """
        obs_ = torch.clone(observations)
        return (obs_ -self._B) * self._W

    # def _reward_around(self, observations:Tensor)->Tensor:
    #     """根据周车和自车的相对状态定义奖励 (越高越好, 不一定是正值)


    #     计算思路:

    #     + 根据 roi (一般取为安全视距) 和周车的距离计算周车的所占的权重, 根据阈值(一般取为超车视距) 将近处的周车权重进一步归一化.
    #     + 根据周车与自车的状态差异计算周车的价值, 并根据权重加权求和得到周车的总价值.

    #     Args:
    #         observations (np.ndarray): shape =  (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]

    #     Returns:
    #         rewards (float): 奖励(越高越好, 正值), (B,1)
    #     """
    #     obs_ =torch.clone(observations)

    #     # 解包数据
    #     ego = obs_[:, 0:1, :]  # 自车数据 (B,1,7)
    #     others = obs_[:, 1:, :]  # 周车数据 (B,15,7)
    #     postion_x = ego[:, 0, 2]  # 自车的 x 坐标 (B,)

    #     # 周车与自车的差值
    #     diff_x = others[:, :, 2] - postion_x.reshape(-1,1)  # 周围车与当前车的 x 坐标差 (前车为正, 后车为负值), (B,15)
    #     diff = others - ego  # 周围车与当前车的状态差 (B,15,7)


    #     # 计算周车的价值
    #     value_cars = -diff @ self.WEIGHT_AROUND # (B,15)


    #     # 根据 diff_x 生成高斯权重 (已按样本归一化), (B,15)
    #     weight = self._distance_weights(diff_x, 200/2000, 500/2000)  # 视野范围: 200m, 权重阈值: 500m

    #     # 逐样本对 15 辆车的价值加权求和: (B,15) * (B,15) 沿最后一维求和 -> (B,1)
    #     value = (value_cars * weight).sum(dim=-1, keepdim=True)

    #     return value

    def _reward_around(self, observations: Tensor) -> Tensor:
        """根据周车和自车的相对状态定义奖励。

        Args:
            observations (Tensor):
                shape = (B, 16, 7)
                7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]
                已经标准化到 0~1。

        Returns:
            Tensor:
                shape = (B, 1)，奖励越高越好，通常为非正值。
        """

        # 自车：(B, 1, 7)
        ego = observations[:, 0:1, :]

        # 周车：(B, 15, 7)
        others = observations[:, 1:, :]

        # 周车与自车的状态差：(B, 15, 7)
        # [v, heading, x, y, ax, ay, lane]
        diff = others - ego

        # 周车与自车的纵向位置差：(B, 15)
        diff_x = diff[:, :, 2]

        # 判断周车位于自车前方还是后方
        #
        # behind_mask: 自车在周车后方，鼓励超车
        # ahead_mask : 自车在周车前方，鼓励甩掉后车
        behind_mask = diff_x < 0
        ahead_mask = ~behind_mask

        # 初始化转换后的差值
        reward_diff = torch.zeros_like(diff)

        # ==========================================================
        # 1. 自车在周车后方：鼓励超越
        # ==========================================================
        #
        # 目标：
        # v       : 自车速度 > 周车速度
        # heading : 航向角差异越大越好
        # x       : diff_x 越小越好
        # y       : diff_y 越大越好
        # ax      : diff_ax 越小越好
        # ay      : diff_ay 越大越好
        # lane    : diff_lane 越大越好
        #
        # 对于需要“越大越好”的量，使用 -diff；
        # 对于需要“越小越好”的量，使用 diff。
        # 由于最终奖励为 -reward_diff @ weight，
        # reward_diff 越小，奖励越高。

        reward_diff[behind_mask, 0] = diff[behind_mask, 0]
        reward_diff[behind_mask, 1] = -torch.abs(diff[behind_mask, 1])
        reward_diff[behind_mask, 2] = diff[behind_mask, 2]
        reward_diff[behind_mask, 3] = -diff[behind_mask, 3]
        reward_diff[behind_mask, 4] = diff[behind_mask, 4]
        reward_diff[behind_mask, 5] = -diff[behind_mask, 5]
        reward_diff[behind_mask, 6] = -diff[behind_mask, 6]

        # ==========================================================
        # 2. 自车在周车前方：鼓励甩掉后车
        # ==========================================================
        #
        # 目标：
        # v       : diff_v 越小越好
        # heading : 航向角差异越小越好
        # x       : diff_x 越小越好
        # y       : diff_y 越大越好
        # ax      : diff_ax 越小越好
        # ay      : diff_ay 越大越好
        # lane    : diff_lane 越大越好

        reward_diff[ahead_mask, 0] = diff[ahead_mask, 0]
        reward_diff[ahead_mask, 1] = torch.abs(diff[ahead_mask, 1])
        reward_diff[ahead_mask, 2] = diff[ahead_mask, 2]
        reward_diff[ahead_mask, 3] = -diff[ahead_mask, 3]
        reward_diff[ahead_mask, 4] = diff[ahead_mask, 4]
        reward_diff[ahead_mask, 5] = -diff[ahead_mask, 5]
        reward_diff[ahead_mask, 6] = -diff[ahead_mask, 6]

        # ==========================================================
        # 3. 根据周车距离计算权重
        # ==========================================================

        weight = self._distance_weights(
            diff_x,
            200 / 2000,
            500 / 2000
        )

        # ==========================================================
        # 4. 根据不同场景的状态差值计算奖励
        # ==========================================================

        value_cars = -reward_diff @ self.WEIGHT_AROUND

        # 逐辆周车加权求和
        value = (value_cars * weight).sum(dim=-1, keepdim=True)

        return value

    def _reward_ego(self, observations:Tensor)->Tensor:
        """根据自车的状态定义奖励 (越高越好, 不一定是正值)

        Args:
            observations (np.ndarray): shape =  (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]

        Returns:
            rewards (float): 奖励(越高越好, 正值)
        """
        obs_ =torch.clone(observations)

        # 解包数据
        ego = obs_[:, 0:1, :]  # 自车数据 (B,7)
        value_ego = -torch.abs(ego - self.GOAL) @ self.WEIGHT_EGO # (B,1)
        return value_ego

    def _reward_by_obs(self, observations:Tensor)->tuple[Tensor, Tensor]:
        """根据状态定义奖励 (越高越好, 不一定是正值)

        *状态已经标准化*

        Args:
            observations (np.ndarray): shape =  (B, 16, 7), 7:[速度, 航向角, x, y, 加速度x, 加速度y, lane]

        Returns:
            rewards (float): 奖励(越高越好)
        """
        # around = self._reward_around(observations)
        ego = self._reward_ego(observations)
        return ego, torch.zeros(ego.shape, dtype=torch.float32)

    def _distance_weights(self, distance: Tensor, roi: float, threshold: float) -> Tensor:
        """根据距离和标准差生成权重

        Args:
            distance (np.ndarray): 各周车与 ego 的距离 (在本车前为正, 本车后为负), shape(B,15)
            roi (float): 视野范围(单位:m), 标准化时为 视野范围/路段长度
            threshold (float): 权重阈值(单位:m), 标准化时为 阈值/路段长度

        Returns:
            各周车的权重 (np.ndarray): shape(B,15)
        """

        sigma = roi / (torch.sqrt(torch.tensor(2.0)) * torch.erfinv(torch.tensor(0.6827)))
        # 计算权重
        weight = torch.exp(-(distance ** 2) / (2 * sigma ** 2))

        # 在 roi 范围内归一化权重
        mask = (distance<=threshold).to(weight.dtype)
        near_sum = (weight * mask).sum(dim=-1, keepdim=True)
        # 防止 near_sum 为 0 时除零
        near_sum = torch.where(near_sum > 0, near_sum, torch.ones_like(near_sum))
        weight = torch.where(mask == 1, weight / near_sum, weight)

        # 避免除以 0 导致的 NaN / Inf (兜底)
        weight = torch.where(torch.isnan(weight) | torch.isinf(weight), 0, weight)
        return weight