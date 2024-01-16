# -*- coding: utf-8 -*-
# @Time : 2023/5/10 11:58
# @Author : yzbyx
# @File : CFModel_ACC.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

from trasim_simplified.core.kinematics.cfm.CFModel_KK import cal_v_safe, CFModel_KK

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM, V_TYPE


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
        self.original_acc = f_param.get("original_acc", True)
        """是否为原始ACC模型，True则代表包含gipps安全速度约束"""
        self.v_safe_dispersed = f_param.get("v_safe_dispersed", True)
        """v_safe计算是否离散化时间"""
        self.a = f_param.get("a", 3.)
        """期望加速度，用于安全速度计算"""
        self.b = f_param.get("b", 3.)
        """期望减速度，用于安全速度计算"""
        self.tau = f_param.get("tau", 1)

        self._s0 = f_param.get("s0", 2)

        self.v0 = f_param.get("v0", 30)

        self.index = None

    def _update_dynamic(self):
        self.dt = self.vehicle.lane.dt
        self.tau = self.dt
        self.gap = self.vehicle.gap
        assert self.original_acc or self.dt == self.tau, print(self.dt, self.tau)
        if self.original_acc:
            self.l_v_a = None
            return
        self._update_v_safe()

    def _update_v_safe(self):
        self.l_v_a = CFModel_KK.update_v_safe(self)

    def get_expect_dec(self):
        return self.b

    def get_expect_acc(self):
        return self.a

    def get_expect_speed(self):
        return self.v0

    def step(self, index, *args):
        self.index = index
        if self.vehicle.leader is None:
            return 3
        self._update_dynamic()
        f_params = [self._k1, self._k2, self._thw, self._s0,
                    self.a, self.b, self.original_acc, self.v_safe_dispersed]
        leader_is_dummy = True if self.vehicle.leader.type == V_TYPE.OBSTACLE else False
        return calculate(*f_params,
                         self.tau, self.gap, self.vehicle.v, self.vehicle.leader.v, self.v0,
                         leader_is_dummy, self.l_v_a)


def calculate(k1_, k2_, thw_, s0_, a, b, original_acc, v_safe_dispersed,
              tau, gap, v, l_v, v_free, leader_is_dummy, l_v_a):
    acc = k1_ * (gap - thw_ * v) + k2_ * (l_v - v)
    if original_acc:
        acc = k1_ * (gap - s0_ - thw_ * v) + k2_ * (l_v - v)
        v_next = v + tau * acc
        if v_next > v_free:
            return (v_free - v) / tau
        elif v_next < 0:
            return - v / tau
        else:
            return acc

    v_c = v + tau * max(- b, min(acc, a))
    v_safe = cal_v_safe(v_safe_dispersed, tau, l_v, gap, b, b)
    if not leader_is_dummy:
        v_safe = min(v_safe, (gap / tau) + l_v_a)
    v_next = max(0, min(v_free, v_c, v_safe))
    return (v_next - v) / tau


def cf_ACC_acc(k1, k2, thw, s0, speed, gap, leaderV, **kwargs):
    return k1 * (gap - s0 - thw * speed) + k2 * (leaderV - speed)


def cf_ACC_equilibrium(thw, s0, speed, **kwargs):
    return s0 + thw * speed


def cf_ACC_acc_module(k1, k2, thw, s0, speed, gap, leaderV, **kwargs):
    k_space = kwargs.get("k_space", 1)
    k_zero = kwargs.get("k_zero", 1)
    return k_space * k1 * (gap - s0 - thw * speed) + k_zero * k2 * (leaderV - speed)


def cf_ACC_equilibrium_module(thw, s0, speed, **kwargs):
    return s0 + thw * speed
