# -*- coding: utf-8 -*-
# @time : 2023/5/10 17:39
# @Author : yzbyx
# @File : CFModel_TPACC.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM, V_TYPE, VehSurr
from trasim_simplified.core.kinematics.cfm.CFModel_KK import cal_v_safe, CFModel_KK


class CFModel_TPACC(CFModel):
    def __init__(self, f_param: dict[str, float]):
        super().__init__()
        self.name = CFM.TPACC
        self.thesis = "Physics of automated driving in framework of three-phase traffic theory (2018)"

        self._v0 = f_param.get("v0", 33.3)

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
        self.tau = f_param.get("tau", 1)
        """仿真步长即反应时间 [s]"""
        self.tau_safe = 1

        self.record_cf_info = f_param.get("record_cf_info", False)
        self.cf_info: Optional[TPACCInfo] = None
        if self.record_cf_info:
            self.cf_info = TPACCInfo()

    @property
    def v_safe_dispersed(self):
        return self._v_safe_dispersed

    def get_expect_dec(self):
        return self._b

    def get_expect_acc(self):
        return self._a

    def get_expect_speed(self):
        return self._v0

    def get_max_speed(self):
        return self.get_speed_limit()

    def get_max_dec(self):
        return 8

    def get_max_acc(self):
        return 5

    def get_com_acc(self):
        return self._a

    def get_com_dec(self):
        return self._b

    def get_safe_s0(self):
        return 2

    def get_time_safe(self):
        return 0.5

    def get_time_wanted(self):
        return 1

    def _update_dynamic(self):
        self.gap = self.gap = self.veh_surr.cp.x - self.veh_surr.ev.x - self.veh_surr.cp.length
        self.dt = self.veh_surr.ev.lane.dt
        self._update_v_safe()

    def _update_v_safe(self):
        # self.l_v_a = CFModel_KK.update_v_safe(self)
        self.l_v_a = self.veh_surr.cp.a

    def step(self, veh_surr: VehSurr):
        self.veh_surr = veh_surr
        if self.veh_surr.ev is None:
            return None
        if self.veh_surr.ev.leader is None:
            speed = max(0., min(self.get_expect_speed(), self.veh_surr.ev.v + self._a * self.veh_surr.ev.dt))
            return (speed - self.veh_surr.ev.v) / self.veh_surr.ev.dt
        self._update_dynamic()
        f_params = [self._kdv, self._k1, self._k2, self._thw, self._g_tau, self._a, self._b, self._v_safe_dispersed]
        leader_is_dummy = True if self.veh_surr.cp.type == V_TYPE.OBSTACLE else False
        result = calculate(
            *f_params, self.dt, self.gap, self.veh_surr.ev.v, self.veh_surr.cp.v, self.get_expect_speed(),
            leader_is_dummy, self.l_v_a, self.tau_safe
        )
        if self.record_cf_info:
            self.cf_info.step.append(self.veh_surr.ev.lane.step_)
            self.cf_info.time.append(self.veh_surr.ev.lane.time_)
            self.cf_info.is_speed_adaptive.append(result[1])
            self.cf_info.is_acc_constraint.append(result[2])
            self.cf_info.is_thw_constraint.append(result[3])
            self.cf_info.is_v_free_constraint.append(result[4])
            self.cf_info.is_v_safe_constraint.append(result[5])

        return result[0]


def calculate(kdv_, k1_, k2_, thw_, g_tau_, acc_, dec_, v_safe_dispersed_,
              dt, gap, v, l_v, v_free, leader_is_dummy, l_v_a, tau=1):
    if gap > v * g_tau_:
        acc = k1_ * (gap - thw_ * v) + k2_ * (l_v - v)
        is_speed_adaptive = 0
    else:
        acc = kdv_ * (l_v - v)
        is_speed_adaptive = 1

    is_acc_constraint = 0 if - dec_ <= acc <= acc_ else 1
    v_c = v + dt * max(- dec_, min(acc, acc_))

    v_safe = cal_v_safe(v_safe_dispersed_, tau, l_v, gap, dec_, dec_)
    is_thw_constraint = 0
    if not leader_is_dummy:
        temp = (gap / tau) + l_v_a
        is_thw_constraint = 1 if v_safe > temp else 0
        v_safe = min(v_safe, temp)

    is_v_free_constraint = 1 if v_free < v_c else 0
    is_v_safe_constraint = 1 if v_safe < v_c else 0
    v_next = max(0, min(v_free, v_c, v_safe))

    a_final = (v_next - v) / dt

    return a_final, \
        is_speed_adaptive, is_acc_constraint, is_thw_constraint, is_v_free_constraint, is_v_safe_constraint


def cf_TPACC_acc(kdv, k1, k2, thw, g_tau, a, b, v_safe_dispersed,
                 interval, gap, speed, leaderV, v_free=30, leader_is_dummy=False, l_v_a=0):
    pass


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
