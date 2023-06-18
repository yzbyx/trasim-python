# -*- coding = uft-8 -*-
# @Time : 2023-04-09 9:44
# @Author : yzbyx
# @File : CFModel_Linear.py
# @Software : PyCharm
from typing import Optional, TYPE_CHECKING

import numpy as np

from trasim_simplified.core.constant import CFM
from trasim_simplified.core.kinematics.cfm import CFModel

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle


class CFModel_Linear(CFModel):
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        # -----模型属性------ #
        self.name = CFM.LINEAR
        self.thesis = ''

        # -----模型变量------ #
        self._T = f_param.get("T", 1.)
        """反应时间 [s]"""
        self._lambda = f_param.get("lambda", 1.)
        """敏感度 [s^-1]"""

        self.v = -1
        self.x = -1
        self.l_v = -1
        self.l_x = -1

    def _update_dynamic(self):
        time = self.vehicle.lane.time_ - self._T
        if time > self._T:
            index = np.where(((time - 1e-4) < np.array(self.vehicle.time_list)) &
                             ((time + 1e-4) > np.array(self.vehicle.time_list)))[0][0]
        else:
            index = 0
        self.pre_v = self.vehicle.speed_list[index]
        self.l_pre_v = self.vehicle.leader.speed_list[index]

    def step(self, *args):
        if self.vehicle.leader is None:
            return 0.
        self._update_dynamic()
        return calculate(self._lambda, self.pre_v, self.l_pre_v)

    def get_expect_dec(self):
        return self.DEFAULT_EXPECT_DEC

    def get_expect_acc(self):
        return self.DEFAULT_EXPECT_ACC

    def get_expect_speed(self):
        return self.get_speed_limit()


def calculate(lambda_, speed, leaderV):
    return lambda_ * (leaderV - speed)
