# -*- coding: utf-8 -*-
# @Time : 2023/5/10 17:39
# @Author : yzbyx
# @File : CFModel_TPACC.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM, V_TYPE
from trasim_simplified.core.kinematics.cfm.CFModel_KK import cal_v_safe, CFModel_KK


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
        self._tau = f_param.get("tau", 1)
        """仿真步长即反应时间 [s]"""

        self._record_cf_info = f_param.get("record_cf_info", False)
        self.cf_info: Optional[TPACCInfo] = None
        if self._record_cf_info:
            self.cf_info = TPACCInfo()

        self.index = None

    @property
    def v_safe_dispersed(self):
        return self._v_safe_dispersed

    def get_expect_dec(self):
        return self._b

    def get_expect_acc(self):
        return self._a

    def get_expect_speed(self):
        return self.get_speed_limit()

    def _update_dynamic(self):
        self.gap = self.vehicle.gap
        self.dt = self.vehicle.lane.dt
        assert self.dt == self._tau
        self._update_v_safe()

    def _update_v_safe(self):
        self.l_v_a = CFModel_KK.update_v_safe(self)

    def step(self, index, *args):
        self.index = index
        if self.vehicle.leader is None:
            return 3
        self._update_dynamic()
        f_params = [self._kdv, self._k1, self._k2, self._thw, self._g_tau, self._a, self._b, self._v_safe_dispersed]
        leader_is_dummy = True if self.vehicle.leader.type == V_TYPE.OBSTACLE else False
        result = calculate(*f_params,
                           self.dt, self.gap, self.vehicle.v, self.vehicle.leader.v, self.get_expect_speed(),
                           leader_is_dummy, self.l_v_a)
        if self._record_cf_info:
            self.cf_info.step.append(self.vehicle.lane.step_)
            self.cf_info.time.append(self.vehicle.lane.time_)
            self.cf_info.is_speed_adaptive.append(result[1])
            self.cf_info.is_acc_constraint.append(result[2])
            self.cf_info.is_thw_constraint.append(result[3])
            self.cf_info.is_v_free_constraint.append(result[4])
            self.cf_info.is_v_safe_constraint.append(result[5])

        return result[0]


def calculate(kdv_, k1_, k2_, thw_, g_tau_, acc_, dec_, v_safe_dispersed_,
              dt, gap, v, l_v, v_free, leader_is_dummy, l_v_a):
    if gap > v * g_tau_:
        acc = k1_ * (gap - thw_ * v) + k2_ * (l_v - v)
        is_speed_adaptive = 0
    else:
        acc = kdv_ * (l_v - v)
        is_speed_adaptive = 1

    is_acc_constraint = 0 if - dec_ <= acc <= acc_ else 1
    v_c = v + dt * max(- dec_, min(acc, acc_))

    v_safe = cal_v_safe(v_safe_dispersed_, dt, l_v, gap, dec_, dec_)
    is_thw_constraint = 0
    if not leader_is_dummy:
        temp = (gap / dt) + l_v_a
        is_thw_constraint = 1 if v_safe > temp else 0
        v_safe = min(v_safe, temp)

    is_v_free_constraint = 1 if v_free < v_c else 0
    is_v_safe_constraint = 1 if v_safe < v_c else 0
    v_next = max(0, min(v_free, v_c, v_safe))

    return (v_next - v) / dt, \
        is_speed_adaptive, is_acc_constraint, is_thw_constraint, is_v_free_constraint, is_v_safe_constraint


class TPACCInfo:
    def __init__(self):
        self.step = []
        """对应时间步"""
        self.time = []
        """对应时间 [s]"""
        self.is_speed_adaptive = []
        """是否处于速度适配区间"""
        self.is_acc_constraint = []
        """计算出的加速度是否被最大加减速限制"""
        self.is_thw_constraint = []
        """安全速度是否被期望时距限制"""
        self.is_v_free_constraint = []
        """vc是否被vFree限制"""
        self.is_v_safe_constraint = []
        """vc是否被vSafe限制"""
