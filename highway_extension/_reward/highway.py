"""高速公路适用的奖励函数.

该版本为 goal-conditioned 版本:
    state = [ego_abs(7), around_rel(15*7), goal(7)]

    ego_abs 特征顺序: [presence, x, y, vx, vy, cos(heading), sin(heading)]
        x 归一化系数: 2000
        y 归一化系数: 12
        vx/vy 归一化系数: 80

    around_rel 沿用 highway_env Kinematics 相对观测:
        x 相对值真实范围约 [-200, 200], 归一化到 [-1, 1]
        y 相对值真实范围约 [-12, 12]
        vx/vy 相对速度真实范围约 [-80, 80]
"""
# region imports
import logging

import numpy as np
import torch
from torch import Tensor, no_grad

from modelsolver.abc.reward import IReward


# endregion
logging.basicConfig(level=logging.INFO, filename="./logs/reward.log", filemode="w", encoding="utf-8")


class HighwayReward(IReward):
    """连续高速公路场景的 goal-conditioned 奖励函数."""

    LANE_WIDTH = 3.7
    LANE_COUNT = 3

    # ego_abs 的绝对归一化系数
    EGO_X_SCALE = 2000.0
    EGO_Y_SCALE = 12.0
    EGO_V_SCALE = 80.0

    # around_rel 的相对归一化系数, 与 highway_env KinematicObservation 默认范围一致.
    REL_X_SCALE = 200.0
    REL_Y_SCALE = 12.0
    REL_V_SCALE = 80.0

    GOAL_X_EPS = 0.005  # 认为“到达 goal.x”的容差, 约 10m

    @property
    def is_learnable(self) -> bool:
        return False

    @no_grad
    def forward(self, **kwargs: Tensor) -> tuple[Tensor, Tensor]:
        if "expert" in kwargs:
            raise NotImplementedError("为 EHER 预留的接口, 还没有实现相关逻辑.")

        state = kwargs["state"]
        next_state = kwargs["next_state"]
        original_done = kwargs.get("original_done", None)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        if original_done is None:
            original_done = torch.zeros((state.shape[0], 1), dtype=torch.float32)
        elif original_done.dim() == 1:
            original_done = original_done.unsqueeze(1)

        ego, around, goal = self._split_state(state)
        next_ego, next_around, next_goal = self._split_state(next_state)

        # 1) 速度奖励: 与 SAC 非 goal 模式中 highway_env 的速度奖励保持同源.
        reward = self._speed_reward(ego)

        # 2) goal-conditioned 稠密奖励: 这是 HER 额外引入的 goal 信号.
        reward = reward + self._goal_reward(ego, goal)

        # 3) 车道保持: 与 SAC 非 goal 模式一致.
        reward = reward + self._lane_keep_reward(ego)

        # 4) 安全: 前车距离 + TTC. 与 SAC 非 goal 模式一致, 但输入改为相对周车.
        reward = reward + self._front_risk_reward(ego, around)

        # 5) 安全: 近距避撞. 与 SAC 非 goal 模式一致.
        reward = reward + self._nearby_vehicle_reward(ego, around)

        # 6) potential-based lane-gap shaping. 与 SAC 非 goal 模式一致.
        reward = reward + 5.0 * (
            0.98 * self._lane_gap_potential(next_ego, next_around)
            - self._lane_gap_potential(ego, around)
        )

        # 7) terminal. 只有原始环境事件才给大额 terminal reward;
        #     HER relabel 出来的 done=1 不再额外给 +100, 避免过度放大激进超车收益.
        done = self._compute_done(ego, goal, original_done)
        actual_success = (original_done > 0).to(torch.float32)
        actual_failure = (original_done < 0).to(torch.float32)
        reward = reward + torch.where(
            actual_success > 0,
            torch.full_like(done, 100.0),
            torch.where(actual_failure > 0, torch.full_like(done, -50.0), torch.zeros_like(done)),
        )

        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return reward, done

    # ------------------------------------------------------------------
    # state 拆分
    # ------------------------------------------------------------------
    def _split_state(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B = state.shape[0]
        ego = state[:, 0:7]
        around = state[:, 7:112].reshape(B, 15, 7)
        goal = state[:, 112:119]
        return ego, around, goal

    # ------------------------------------------------------------------
    # 各奖励分量
    # ------------------------------------------------------------------
    def _speed_reward(self, ego: Tensor) -> Tensor:
        vx = (ego[:, 3] * self.EGO_V_SCALE).clamp(min=0.0)
        return (vx / 30.0).clamp(max=1.0).unsqueeze(1)

    def _goal_reward(self, ego: Tensor, goal: Tensor) -> Tensor:
        """HER 的 goal 信号.

        为避免 HER 变得比 SAC 更激进, 该项权重保持较小; 安全项才是主要约束.
        """
        dx = (ego[:, 1] - goal[:, 1]) * self.EGO_X_SCALE
        dy = (ego[:, 2] - goal[:, 2]) * self.EGO_Y_SCALE
        dvx = (ego[:, 3] - goal[:, 3]) * self.EGO_V_SCALE

        reward = (
            -0.1 * torch.abs(dx) / self.EGO_X_SCALE
            - 0.1 * torch.abs(dy) / self.EGO_Y_SCALE
            - 0.05 * torch.abs(dvx) / self.EGO_V_SCALE
        )
        return reward.unsqueeze(1)

    def _lane_keep_reward(self, ego: Tensor) -> Tensor:
        y_position = ego[:, 2] * self.EGO_Y_SCALE
        lane_id = ((y_position - self.LANE_WIDTH / 2.0) / self.LANE_WIDTH).round().int()
        lane_id = lane_id.clamp(0, self.LANE_COUNT - 1)
        center_y = (lane_id.to(torch.float32) + 0.5) * self.LANE_WIDTH
        lat = y_position - center_y

        penalty = torch.min((lat / self.LANE_WIDTH) ** 2, torch.full_like(lat, 4.0))
        return (-2.0 * penalty).unsqueeze(1)

    def _front_risk_reward(self, ego: Tensor, around: Tensor) -> Tensor:
        B = ego.shape[0]
        ego_y = ego[:, 2] * self.EGO_Y_SCALE
        ego_vx = ego[:, 3] * self.EGO_V_SCALE

        rewards = torch.zeros(B, 1, dtype=torch.float32)
        for i in range(B):
            forward_speed = max(float(ego_vx[i].item()), 0.0)
            safe_gap = 8.0 + 1.5 * forward_speed

            front_gap = float("inf")
            lead_vx = 0.0
            for j in range(around.shape[1]):
                if float(around[i, j, 0].item()) < 0.5:
                    continue
                dx = float(around[i, j, 1].item()) * self.REL_X_SCALE
                if dx <= 0:
                    continue
                dy = float(around[i, j, 2].item()) * self.REL_Y_SCALE
                if abs(dy) < self.LANE_WIDTH and dx < front_gap:
                    front_gap = dx
                    lead_vx = float(ego_vx[i].item()) + float(around[i, j, 3].item()) * self.REL_V_SCALE

            if np.isfinite(front_gap):
                if front_gap < safe_gap:
                    ratio = front_gap / safe_gap
                    rewards[i, 0] -= 2.5 * (1.0 - ratio) ** 2

                closing_speed = max(forward_speed - lead_vx, 0.0)
                ttc = front_gap / max(closing_speed, 1e-3)
                if ttc < 2.0:
                    rewards[i, 0] -= 3.0 * (1.0 - ttc / 2.0) ** 2

        return rewards

    def _nearby_vehicle_reward(self, ego: Tensor, around: Tensor, radius: float = 20.0) -> Tensor:
        B = ego.shape[0]
        rewards = torch.zeros(B, 1, dtype=torch.float32)
        for i in range(B):
            penalty = 0.0
            for j in range(around.shape[1]):
                if float(around[i, j, 0].item()) < 0.5:
                    continue
                dx = float(around[i, j, 1].item()) * self.REL_X_SCALE
                if dx < -10.0:
                    continue
                dy = float(around[i, j, 2].item()) * self.REL_Y_SCALE
                dist = float(np.hypot(dx, dy))
                if dist < radius:
                    penalty += (1.0 - dist / radius)
            rewards[i, 0] -= 0.6 * min(penalty, 2.0)

        return rewards

    def _lane_gap_potential(self, ego: Tensor, around: Tensor) -> Tensor:
        B = ego.shape[0]
        ego_x = ego[:, 1] * self.EGO_X_SCALE
        ego_y = ego[:, 2] * self.EGO_Y_SCALE
        ego_vx = ego[:, 3] * self.EGO_V_SCALE

        phi = torch.zeros(B, 1, dtype=torch.float32)
        for i in range(B):
            forward_speed = max(float(ego_vx[i].item()), 0.0)
            safe_gap = 8.0 + 1.5 * forward_speed

            lane_gaps = [float("inf")] * self.LANE_COUNT
            for j in range(around.shape[1]):
                if float(around[i, j, 0].item()) < 0.5:
                    continue
                dx = float(around[i, j, 1].item()) * self.REL_X_SCALE
                if dx <= 0:
                    continue
                dy = float(around[i, j, 2].item()) * self.REL_Y_SCALE
                other_y = float(ego_y[i].item()) + dy
                lane_id = int(np.clip(round((other_y - self.LANE_WIDTH / 2.0) / self.LANE_WIDTH), 0, self.LANE_COUNT - 1))
                lane_gaps[lane_id] = min(lane_gaps[lane_id], dx)

            safe_gaps = [g for g in lane_gaps if np.isfinite(g) and g > safe_gap]
            if not safe_gaps:
                phi[i, 0] = 0.0
            else:
                best_gap = max(safe_gaps)
                if not np.isfinite(best_gap):
                    phi[i, 0] = 1.0
                else:
                    phi[i, 0] = best_gap / (best_gap + 30.0)

        return phi

    # ------------------------------------------------------------------
    # termination
    # ------------------------------------------------------------------
    def _compute_done(self, ego: Tensor, goal: Tensor, original_done: Tensor) -> Tensor:
        goal_reached = (ego[:, 1] >= goal[:, 1] - self.GOAL_X_EPS).unsqueeze(1).to(torch.float32)
        success = ((original_done > 0).to(torch.float32) + goal_reached).clamp(max=1.0)

        done = torch.where(
            original_done < 0,
            torch.full_like(original_done, -1.0),
            torch.where(success > 0, torch.ones_like(original_done), torch.zeros_like(original_done)),
        )
        return done
