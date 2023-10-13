# -*- coding = uft-8 -*-
# @Time : 2022-04-04 21:48
# @Author : yzbyx
# @File : CFModel_NonLinearGHR.py
# @Software : PyCharm
from typing import Optional, TYPE_CHECKING

import numba
import numpy as np

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle


class CFModel_NonLinearGHR(CFModel):
    """
    m : 速度系数

    l : 间距系数

    a : 与敏感度相关的系数

    tau : 反应时间
    """

    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        self.name = CFM.NON_LINEAR_GHR
        self.thesis = 'Nonlinear Follow-The-Leader Models of Traffic Flow (1961)'

        self._m = f_param.get("m", 1)
        """速度系数"""
        self._l = f_param.get("l", 1)
        """间距系数"""
        self._a = f_param.get("a", 44.1 / 3.6)
        """与敏感度相关的系数"""
        self._tau = f_param.get("tau", 1.5)
        """反应时间"""

    def _update_dynamic(self):
        time = self.vehicle.lane.time_ - self._tau
        index = np.where(((time - 1e-4) < np.array(self.vehicle.time_list)) &
                         ((time + 1e-4) > np.array(self.vehicle.time_list)))[0][0]
        self.pre_x = self.vehicle.pos_list[index]
        self.pre_v = self.vehicle.speed_list[index]
        self.l_pre_x = self.pre_x + self.vehicle.dhw_list[index]
        self.l_pre_v = self.vehicle.leader.speed_list[index]

    def step(self, index, *args):
        time = self.vehicle.lane.time_ - self._tau
        if time <= self._tau:
            if len(self.vehicle.acc_list) != 0:
                return self.vehicle.acc_list[-1]
            else:
                return self.vehicle.a
        if self.vehicle.leader is None:
            return self.get_expect_acc()
        self._update_dynamic()
        return calculate(self.vehicle.v, self.pre_x, self.l_pre_x, self.pre_v, self.l_pre_v)

    def get_expect_dec(self):
        return self.DEFAULT_EXPECT_DEC

    def get_expect_acc(self):
        return self.DEFAULT_EXPECT_ACC

    def get_expect_speed(self):
        return self.get_speed_limit()


@numba.njit()
def calculate(a, m, l, speed, pre_x, l_pre_x, pre_v, l_pre_v) -> float:
    return a * (speed ** m) * (l_pre_v - pre_v) / ((l_pre_x - pre_x) ** l)
