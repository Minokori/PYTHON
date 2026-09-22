"""连续高速公路的一些实用函数。主要用于转换动作/状态/观测的形状和标准化/反标准化"""
import numpy as np


config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 16,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False, # True: 观测值是绝对坐标, False: 观测值是相对坐标
                "normalize": True,  # 归一化, 目前使用了默认值[100, 100, 20, 20].
                "clip": True,
                "see_behind": False,
                "observe_intentions": False,
                "include_obstacles": False
            },
            "action": {
                "type": "ContinuousAction",
            },
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 50,
            "policy_frequency": 5,
            "duration": 400.0,
            "screen_width": 1200,
            "screen_height": 400,
            "scaling": 7.0,
            "centering_position": [0.5, 0.5],
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False,
            "neighbour_vehicles_connected_lanes": True,
            "add_walls":True,
            "right_lane_reward":0,
            "high_speed_reward":1.0,
            "collision_reward":-1.0,
            "reward_speed_range":[0,30],
            "normalize_reward":False

        }



