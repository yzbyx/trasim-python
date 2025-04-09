# -*- coding: utf-8 -*-
# @Time : 2025/3/22 20:55
# @Author : yzbyx
# @File : agent.py
# Software: PyCharm
import abc
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.agent.traj_predictor import TrajPred, get_pred_net
from trasim_simplified.core.agent.vehicle import Vehicle
from trasim_simplified.core.constant import RouteType, VehSurr
from trasim_simplified.core.kinematics.cfm import CFModel

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class AgentBase(Vehicle, abc.ABC):
    """
    自主驾驶类
    """
    def __init__(self, lane: 'LaneAbstract', type_: int, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.crashed = False

        self.state = None
        self.lc_end_step = -1
        self.lc_step_conti = 0

        self.target_speed = 30

        self.pred_net: Optional[TrajPred] = get_pred_net()

    def get_state_for_traj(self):
        """返回车辆状态量[x, dx, ddx, y, dy, ddy]"""
        return np.array([self.x, self.v, self.a, self.y, self.v_lat, self.a_lat])

    def lc_intention_judge(self):
        """判断是否换道并计算换道轨迹"""
        self.target_lane, _ = self.lc_model.step(self.pack_veh_surr())

    def lc_decision_making(self):
        pass

    def cal_vehicle_control(self):
        next_acc_block = np.inf
        if self.lane.index not in self.destination_lane_indexes:
            next_acc_block = self.cf_model.step(VehSurr(ev=self, cp=self.lane.road.end_weaving_block_veh))
        next_acc = self.cf_model.step(self.pack_veh_surr())
        self.next_acc = min(next_acc, next_acc_block)
        self.next_delta = self.cf_lateral_control()
        return self.next_acc, self.next_delta

    def pred_risk(self):
        """预测周边车辆轨迹、判断碰撞概率"""

