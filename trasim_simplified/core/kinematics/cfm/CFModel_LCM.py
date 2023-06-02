# -*- coding: utf-8 -*-
# @Time : 2023/6/1 11:11
# @Author : yzbyx
# @File : CFModel_LCM.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_LCM(CFModel):
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        self.name = CFM.LCM
        self.thesis = "Vehicle Longitudinal Control and Traffic Stream Modeling (Ni, Leonard, Jia, & Wang, 2015)"

        self._a = f_param.get("a", 4)
        """启动加速度 [m/s^2]"""
        self._b = f_param.get("b", 9)
        """紧急情况下当前车认为的减速度 [m/s^2]"""
        self._gamma = f_param.get("gamma", 0.5 * (1 / 9 - 1 / 6))
        """当前车的激进系数 [s^2/m]"""
        self._v0 = f_param.get("v0", 30)
        """期望速度"""
        self._tau = f_param.get("tau", 1)
        """驾驶员的感知反应时间 [s]"""

    def get_expect_dec(self):
        return self._b

    def get_expect_acc(self):
        return self._a

    def get_expect_speed(self):
        return self._v0

    def _update_dynamic(self):
        self.gap = self.vehicle.gap

    def step(self, index, *args):
        if self.vehicle.leader is None:
            return 0
        self._update_dynamic()
        f_params = [self._a, self._b, self._gamma, self._v0, self._tau]
        return calculate(*f_params, self.gap, self.vehicle.v, self.vehicle.leader.v, self.vehicle.leader.length)


def calculate(a_, b_, gamma_, v0_, tau_, gap, v, l_v, l_length):
    """

    :param a_:
    :param b_:
    :param gamma_: 激进系数
    :param tau_:
    :param v0_:
    :param gap:
    :param v:
    :param l_v:
    :param l_length: 有效车长
    :return:
    """
    l_b = 1 / (1 / b_ - 2 * gamma_)
    s_star = (v ** 2) / (2 * b_) - (l_v ** 2) / (2 * l_b) + tau_ * v + l_length
    a = a_ * (1 - (v / v0_) - np.exp(1 - (gap / s_star)))
    return a
