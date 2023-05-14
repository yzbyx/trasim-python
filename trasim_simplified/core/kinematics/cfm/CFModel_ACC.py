# -*- coding: utf-8 -*-
# @Time : 2023/5/10 11:58
# @Author : yzbyx
# @File : CFModel_ACC.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

from trasim_simplified.core.kinematics.cfm.CFModel_KK import cal_v_safe

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_ACC(CFModel):
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        self.name = CFM.ACC
        self.thesis = "Modeling cooperative and autonomous adaptive" \
                      " cruise control dynamic responses using experimental data (2014)"
        self._k1 = f_param.get("k1", 0.3)
        """期望间距差系数"""
        self._k2 = f_param.get("k2", 0.3)
        """速度差系数"""
        self._thw = f_param.get("thw", 1.3)  # 原文似乎未提供
        """期望车头时距"""
        self._original_acc = f_param.get("original_acc", True)
        """是否为原始ACC模型，True则代表包含gipps安全速度约束"""
        self._v_safe_dispersed = f_param.get("v_safe_dispersed", True)
        """v_safe计算是否离散化时间"""
        self._a = f_param.get("a", 3.)
        """期望加速度，用于安全速度计算"""
        self._b = f_param.get("b", 3.)
        """期望减速度，用于安全速度计算"""

    def get_expect_dec(self):
        return self._b

    def get_expect_acc(self):
        return self._a

    def get_expect_speed(self):
        return self.vehicle.lane.speed_limit

    def _update_dynamic(self):
        self.gap = self.vehicle.gap

    def step(self, index, *args):
        if self.vehicle.leader is None:
            return 0.
        self._update_dynamic()
        f_params = [self._k1, self._k2, self._thw, self._a, self._b, self._original_acc, self._v_safe_dispersed]
        return calculate(*f_params, self.vehicle.leader.cf_model.get_expect_dec(),
                         self.dt, self.gap, self.vehicle.v, self.vehicle.leader.v, self.get_expect_speed())


def calculate(k1_, k2_, thw_, acc_, dec_, original_acc_, v_safe_dispersed_,
              l_dec_, dt, gap, v, l_v, v_free):
    acc = k1_ * (gap - thw_ * v) + k2_ * (l_v - v)
    if original_acc_:
        return acc

    v_c = v + dt * max(- dec_, min(acc, acc_))
    v_safe = cal_v_safe(v_safe_dispersed_, dt, l_v, gap, dec_, l_dec_)
    v_next = max(0, min(v_free, v_c, v_safe))
    return (v_next - v) / dt
