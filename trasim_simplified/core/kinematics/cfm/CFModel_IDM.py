# -*- coding = uft-8 -*-
# @time : 2022-04-04 14:22
# @Author : yzbyx
# @File : CFModel_IDM.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Optional
import numba
import numpy as np

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM, VehSurr


class CFModel_IDM(CFModel):
    """
    'v0': 33.3,     # 期望速度

    's0': 2,        # 静止安全间距

    's1': 0,        # 与速度相关的安全距离参数

    'delta': 4,     # 加速度指数

    'T': 1.6,       # 安全车头时距

    'omega': 0.73,  # 最大加速度

    'd': 1.67       # 期望减速度
    """

    def __init__(self, f_param: dict[str, float]):
        super().__init__()
        # -----模型属性------ #
        self.name = CFM.IDM
        self.thesis = 'Congested traffic states in empirical observations and microscopic simulations (2000)'

        # -----模型变量------ #
        self._v0 = f_param.get("v0", 33.3)
        """期望速度"""
        self._s0 = f_param.get("s0", 2)
        """静止安全间距"""
        self._s1 = f_param.get("s1", 0)
        """与速度相关的安全距离参数"""
        self._delta = f_param.get("delta", 4)
        """加速度指数"""
        self._T = f_param.get("T", 1.6)
        """安全车头时距"""
        self._omega = f_param.get("omega", 0.73)
        """舒适加速度"""
        self._d = f_param.get("d", 1.67)
        """舒适减速度"""

        self._time_safe = f_param.get("time_safe", 1)

    def _update_dynamic(self):
        self.gap = self.veh_surr.cp.x - self.veh_surr.ev.x - self.veh_surr.cp.length
        self.dt = self.veh_surr.ev.dt

    def step(self, veh_surr: VehSurr):
        """
        计算下一时间步的加速度

        :return: 下一时间步的加速度
        """
        self.veh_surr = veh_surr
        if self.veh_surr.ev is None:
            return None
        if self.veh_surr.cp is None:
            expect_speed = self.get_expect_speed()
            expect_acc = self.get_expect_acc()
            acc = min(expect_acc, (expect_speed - self.veh_surr.ev.v) / self.veh_surr.ev.dt)
            return acc
        self._update_dynamic()
        T_wanted = self._T
        if self.veh_surr.ev.is_gaming:
            T_wanted = self.veh_surr.ev.game_factor
        return cf_IDM_acc_jit(self._s0, self._s1, min(self._v0, self.get_speed_limit()), T_wanted,
                              self._omega, self._d,
                              self._delta, self.veh_surr.ev.v, self.gap, self.veh_surr.cp.v)

    def equilibrium_state(self, speed, dhw, v_length):
        """
        通过平衡态速度计算三参数

        :param dhw: 平衡间距
        :param v_length: 车辆长度
        :param speed: 平衡态速度
        :return: KQV三参数的值
        """
        sStar = self._s0 + self._s1 * np.sqrt(speed / self._v0) + self._T * speed
        dhw = sStar / np.sqrt(1 - np.power(speed / self._v0, self._delta)) + v_length
        k = 1 / dhw
        v = speed
        q = k * v
        return {"K": k, "Q": q, "V": v}

    def basic_diagram_k_to_q(self, dhw, car_length, speed_limit=None):
        import sympy
        if speed_limit is not None:
            v0 = speed_limit
        else:
            v0 = self._v0
        v = sympy.symbols("v", real=True)
        expr = self._omega * (1 - (v / v0) ** self._delta -
                              ((self._s0 + self._s1 * sympy.sqrt(v / v0) + self._T * v) / (dhw - car_length)) ** 2)
        res: list[float] = sympy.solve(expr, v)
        res.sort()
        return res[-1] * 3.6

    def get_jam_density(self, car_length):
        return 1 / (self._s0 + car_length)

    def get_expect_dec(self):
        return self._d

    def get_expect_acc(self):
        return self._omega

    def get_expect_speed(self):
        return self._v0

    def get_speed_limit(self):
        return self._v0

    def get_max_dec(self):
        return 8

    def get_max_acc(self):
        return 5

    def get_com_acc(self):
        return self._omega

    def get_com_dec(self):
        return self._d

    def get_safe_s0(self):
        return self._s0

    def get_time_safe(self):
        return self._time_safe

    def get_time_wanted(self):
        return self._T


# @numba.njit()
def cf_IDM_acc_jit(s0, s1, v0, T, omega, d, delta, speed, gap, leaderV) -> dict:
    sStar = s0 + s1 * np.sqrt(speed / v0) + T * speed + speed * (speed - leaderV) / (2 * np.sqrt(omega * d))
    # sStar = s0 + np.max(np.array([0, s1 * np.sqrt(speed / v0) +
    #                               T * speed + speed * (speed - leaderV) / (2 * np.sqrt(omega * d))]))
    # 计算车辆下一时间步加速度
    return omega * (1 - np.power(speed / v0, delta) - np.power(sStar / gap, 2))


def cf_IDM_acc(s0, v0, T, omega, d, delta, speed, gap, leaderV, **kwargs) -> dict:
    return cf_IDM_acc_jit(s0, 0, v0, T, omega, d, delta, speed, gap, leaderV)


# @numba.njit()
def cf_IDM_equilibrium_jit(s0, s1, v0, T, delta, v):
    return (s0 + v * T + s1 * np.sqrt(v / v0)) / np.sqrt(1 - np.power(v / v0, delta))


def cf_IDM_equilibrium(s0, v0, T, delta, speed, **kwargs):
    return cf_IDM_equilibrium_jit(s0, 0, v0, T, delta, speed)


def cf_IDM_acc_module(s0, v0, T, omega, d, delta, speed, gap, leaderV, **kwargs):
    k_speed = kwargs.get("k_speed", 1)
    k_space = kwargs.get("k_space", 1)
    k_zero = kwargs.get("k_zero", 1)
    # sStar = (np.max([k_space * s0, k_space * (s0 + T * speed +
    #          k_zero * speed * (speed - leaderV) / (2 * np.sqrt(omega * d)))]))
    sStar = k_space * (s0 + T * speed) + k_zero * speed * (speed - leaderV) / (2 * np.sqrt(omega * d))
    # if T * speed + k_zero * speed * (speed - leaderV) / (2 * np.sqrt(omega * d)) < 0:
    #     print("delta v part < 0")
    return k_speed * omega * (1 - np.power(speed / v0, delta)) - omega * np.power(sStar / gap, 2)


def cf_IDM_equilibrium_module(s0, v0, T, delta, speed, **kwargs):
    k_speed = kwargs.get("k_speed", 1)
    k_space = kwargs.get("k_space", 1)
    return k_space * (s0 + speed * T) / np.sqrt(k_speed * (1 - np.power(speed / v0, delta)))


if __name__ == '__main__':
    print(cf_IDM_acc(2, 0, 30, 1.6, 0.73, 1.67, 4, 0, 7.5, 3.2))
    cf = CFModel_IDM(None, {})
    print(cf.basic_diagram_k_to_q(10, 5))
