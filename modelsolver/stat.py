"""强化学习统计数据类型定义"""
from typing import TypedDict


class AgentStatistics(TypedDict):
    """强化学习统计数据"""

    actor_loss:list[float]
    """Actor 损失."""
    critic_loss:list[float]
    """Critic1 损失"""
    critic_other_loss:list[float]
    """Critic2 损失"""
    alpha_loss:list[float]
    """Alpha 损失"""
    alpha:list[float]
    """Alpha 温度值"""

    td_error:list[float]
    """TD误差均值"""
    td_other_error:list[float]
    """TD_other误差均值"""
    q_value_mean:list[float]
    """Q值均值"""
    q_other_value_mean:list[float]
    """Q_other值均值"""
    episode_id:list[int]
    """当前 episode 的 ID"""

    episode_return:list[float]
    """当前 episode 的累计奖励"""
    success:list[float]
    """当前 episode 是否成功"""

    eval_mean_return:list[float]
    """评估的平均回报

    *例如每 5k 或 10k env steps 评估一次，每次跑 5~10 个无探索噪声episode*
    """

    eval_std_return:list[float]
    """评估的回报标准差"""

    eval_success_rate:list[float]
    """评估的成功率"""

    eval_mean_episode_length:list[float]
    """评估的平均 episode 长度"""
    global_env_step:list[int]
    """全局环境交互步数"""




