# -*- coding: utf-8 -*-
# @Time : 2023/5/27 10:52
# @Author : yzbyx
# @File : CFModel_CACC.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_CACC(CFModel):
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        self.name = CFM.CACC
        self.thesis = "Cooperative adaptive cruise control in real traffic situations (2014)"

        self._k_p = f_param.get("k_p", 0.45)
        """间距误差系数"""
        self._k_d = f_param.get("k_d", 0.25)
        """间距误差变化率系数"""
        self._thw = f_param.get("thw", 0.6)
        """期望净间距时距(去除停车净间距) [s]"""
        self._a = f_param.get("a", 1)
        """最大加速度 [m/s^2]"""
        self._b = f_param.get("b", 2.8)
        """最大减速度 [m/s^2]"""
        self._s0 = f_param.get("s0", 2)
        """最短净间距 [m]"""

        self.pre_e = None

    def get_expect_dec(self):
        return self._b

    def get_expect_acc(self):
        return self._a

    def get_expect_speed(self):
        return self.get_speed_limit()

    def _update_dynamic(self):
        self.dt = self.vehicle.lane.dt

    def step(self, index, *args):
        if self.vehicle.leader is None:
            return 0.
        self._update_dynamic()
        f_params = [self._k_p, self._k_d, self._thw, self._a, self._b, self._s0]
        acc, e = calculate(*f_params, self.dt, self.vehicle.gap, self.vehicle.v,
                           self.vehicle.a, self.vehicle.leader.v, self.pre_e)
        # self.pre_e = e
        return acc


def calculate(k_p_, k_d_, thw_, a_, b_, s0_, dt, gap, v, a, l_v):
    # e_k = gap - thw_ * v - s0_
    # if pre_e is None:
    #     pre_e = e_k
    # v_k = v + k_p_ * e_k + k_d_ * (e_k - pre_e) / dt

    # v_k = (v + k_p_ * gap + ((l_v - v) - thw_ * a) * k_d_) / (1 + k_p_ * thw_)

    e_k = gap - thw_ * v - s0_
    speed_error = l_v - v - thw_ * a
    v_k = v + e_k * k_p_ + speed_error * k_d_

    a_k = (v_k - v) / dt
    a_k_final = max(- b_, min(a_k, a_))
    return a_k_final, e_k
