# -*- coding = uft-8 -*-
# @Time : 2023-04-09 9:44
# @Author : yzbyx
# @File : CFModel_Linear.py
# @Software : PyCharm
from typing import Optional, TYPE_CHECKING

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

    def _update_dynamic(self):
        assert self.dt == self._T

    def step(self, *args):
        f_param = [self._T, self._lambda]
        if args:
            return calculate(*f_param, *args)
        else:
            return calculate(*f_param, self.vehicle.dynamic["speed"], self.vehicle.dynamic["xOffset"],
                             self.vehicle.leader.dynamic["speed"], self.vehicle.leader.dynamic["xOffset"],
                             self.vehicle.leader.static["length"])

def calculate(T, lambda_, speed, xOffset, leaderV, leaderX, leaderL):
    return lambda_ * (leaderV - speed)