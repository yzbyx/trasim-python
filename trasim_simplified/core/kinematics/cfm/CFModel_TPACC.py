# -*- coding: utf-8 -*-
# @Time : 2023/5/10 17:39
# @Author : yzbyx
# @File : CFModel_TPACC.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM
from trasim_simplified.core.kinematics.cfm.CFModel_KK import cal_v_safe


class CFModel_TPACC(CFModel):
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        self.name = CFM.TPACC
        self.thesis = "Physics of automated driving in framework of three-phase traffic theory (2018)"

        self._kdv = f_param.get("kdv", 0.3)
        """速度适配区间的速度差系数"""
        self._k1 = f_param.get("k1", 0.3)
        """期望间距差系数"""
        self._k2 = f_param.get("k2", 0.3)
        """速度差系数"""
        self._thw = f_param.get("thw", 1.3)  # 原文似乎未提供
        """期望时距，从公式上看并非车头时距"""
        self._g_tau = f_param.get("g_tau", 1.4)
        """速度适配时距"""
        self._a = f_param.get("a", 3.)
        """期望加速度，用于安全速度计算"""
        self._b = f_param.get("b", 3.)
        """期望减速度，用于安全速度计算"""
        self._v_safe_dispersed = f_param.get("v_safe_dispersed", True)
        """v_safe计算是否离散化时间"""

    def get_expect_dec(self):
        return self._b

    def get_expect_acc(self):
        return self._a

    def get_expect_speed(self):
        return self.vehicle.lane.speed_limit

    def _update_dynamic(self):
        self.gap = self.vehicle.gap
        self.dt = self.vehicle.lane.dt

    def step(self, index, *args):
        if self.vehicle.leader is None:
            return 0.
        self._update_dynamic()
        f_params = [self._kdv, self._k1, self._k2, self._thw, self._g_tau, self._a, self._b, self._v_safe_dispersed]
        return calculate(*f_params, self.vehicle.leader.cf_model.get_expect_dec(),
                         self.dt, self.gap, self.vehicle.v, self.vehicle.leader.v, self.get_expect_speed())


def calculate(kdv_, k1_, k2_, thw_, g_tau_, acc_, dec_, v_safe_dispersed, l_dec_, dt, gap, v, l_v, v_free):
    if gap > v * g_tau_:
        acc = k1_ * (gap - thw_ * v) + k2_ * (l_v - v)
    else:
        acc = kdv_ * (l_v - v)
    v_c = v + dt * max(- dec_, min(acc, acc_))

    v_safe = cal_v_safe(v_safe_dispersed, dt, l_v, gap, dec_, l_dec_)
    v_next = max(0, min(v_free, v_c, v_safe))
    return (v_next - v) / dt
