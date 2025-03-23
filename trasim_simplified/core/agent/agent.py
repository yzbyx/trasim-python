# -*- coding: utf-8 -*-
# @Time : 2025/3/22 20:55
# @Author : yzbyx
# @File : agent.py
# Software: PyCharm
import abc
from typing import Union, TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.agent.vehicle import Vehicle
from trasim_simplified.core.kinematics.cfm import get_cf_model, CFModel
from traj_predictor.dl_train import

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class AgentBase(Vehicle, abc.ABC):
    """
    自主驾驶类
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crashed = False
        self.action = None

        self.PREVIEW_TIME = 1
        self.MIN_PREVIEW_S = 2
        self.target_lane: 'LaneAbstract' = self.lane
        self.state = None
        self.lc_end_step = -1
        self.lc_step_conti = 0
        self.lc_direction = 0
        self.lane_changing = True

        self.cf_model: Optional[CFModel] = None

    def act(self, action: Union[dict, str] = None):
        if self.crashed:
            return
        self.state = self.get_state()
        # 评估换道
        self.lane_change_judge()

        if self.lane.step_ >= self.lc_end_step:
            self.lane_changing = False
            self.lc_step_conti = 0
            self.lc_direction = 0
            self.target_lane = self.lane

            delta = self.cf_lateral_control()
            acc = self.cf_longitude_control()
        else:
            self.lane_changing = True
            # 横向控制
            opti_v, opti_delta = self.lc_lateral_control()
            delta, v = opti_delta[0], opti_v[0]
            # 纵向控制
            acc = self.lc_longitude_control(v)
            self.lc_step_conti += 1

        Vehicle.act(
            self, {"acceleration": acc, "steering": delta}
        )

    def get_state(self):
        """返回车辆状态量[x, dx, ddx, y, dy, ddy]"""
        return np.array([self.x, self.v, self.a, self.y, self.v_lat, self.a_lat])

    @abc.abstractmethod
    def lane_change_judge(self):
        """判断是否换道并计算换道轨迹"""
        pass

    def cf_lateral_control(self):
        """自动驾驶跟驰过程/人类驾驶全过程的预瞄横向控制"""
        preview_s = max(self.MIN_PREVIEW_S, self.v * self.PREVIEW_TIME)
        l_lat = self.target_lane.y_center - self.y
        dist = np.sqrt(preview_s ** 2 + l_lat ** 2)  # 车头到预瞄点的距离
        theta = np.arctan2(l_lat, preview_s)

        R_h = (2 * self.l_rear_axle_2_head + dist) / (2 * (theta + self.lane.heading - self.yaw))
        R_r = np.sqrt(R_h ** 2 - self.l_rear_axle_2_head ** 2)

        delta = np.arctan2(self.wheelbase, R_r)

        d_delta = np.clip(delta - self.delta, -self.D_DELTA_MAX * self.dt, self.D_DELTA_MAX * self.dt)
        delta = self.delta + d_delta
        delta = np.clip(delta, -self.DELTA_MAX, self.DELTA_MAX)
        return delta

    def cf_longitude_control(self):
        """自动驾驶与人类驾驶公用的纵向跟踪控制"""
        acc = self.cf_model.step(self.lane.step_)
        return acc

    def lc_lateral_control(self):
        """横向控制"""
        pass

    def lc_longitude_control(self):
        pass

    def pred_risk(self):
        """预测周边车辆轨迹、判断碰撞概率"""

    def cal_lc_direction(self):
        """计算换道方向"""
        if self.lane_changing:
            return self.lc_direction

        pass

