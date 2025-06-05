# -*- coding: utf-8 -*-
# @time : 2024/1/19 21:05
# @Author : yzbyx
# @File : CFM_IDM_VS.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional
import numba
import numpy as np

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_IDM_VS(CFModel):
    """
    只包含IDM模型中的期望速度以及期望间距模块
    """
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        # -----模型属性------ #
        self.name = CFM.IDM
        self.thesis = 'Congested traffic states in empirical observations and microscopic simulations (2000)'

        # -----模型变量------ #
        self._v0 = f_param.get("v0", 33.3)
        """期望速度"""
        self._s0 = f_param.get("s0", 2)
        """静止安全间距"""
        self._delta = f_param.get("delta", 4)
        """加速度指数"""
        self._T = f_param.get("T", 1.6)
        """安全车头时距"""
        self._omega = f_param.get("omega", 0.73)
        """舒适加速度"""

    def _update_dynamic(self):
        pass

    def step(self, *args):
        """
        计算下一时间步的加速度

        :param args: 为了兼容矩阵计算设置的参数直接传递
        :return: 下一时间步的加速度
        """
        if self.vehicle.leader is None:
            return self.get_expect_acc()
        self._update_dynamic()
        return cf_IDM_VS_acc_jit(self._s0, min(self._v0, self.get_speed_limit()), self._T, self._omega,
                                 self._delta, self.vehicle.v, self.vehicle.gap)

    def get_expect_dec(self):
        return None

    def get_expect_acc(self):
        return self._omega

    def get_expect_speed(self):
        return self._v0


@numba.njit()
def cf_IDM_VS_acc_jit(s0, v0, T, omega, delta, speed, gap) -> dict:
    sStar = s0 + T * speed
    # sStar = s0 + np.max(np.array([0, s1 * np.sqrt(speed / v0) +
    #                               T * speed + speed * (speed - leaderV) / (2 * np.sqrt(omega * d))]))
    # 计算车辆下一时间步加速度
    return omega * (1 - np.power(speed / v0, delta) - np.power(sStar / gap, 2))


def cf_IDM_VS_acc(s0, v0, T, omega, delta, speed, gap, **kwargs) -> dict:
    return cf_IDM_VS_acc_jit(s0, v0, T, omega, delta, speed, gap)


@numba.njit()
def cf_IDM_VS_equilibrium_jit(s0, v0, T, delta, v):
    return (s0 + v * T) / np.sqrt(1 - np.power(v / v0, delta))


def cf_IDM_VS_equilibrium(s0, v0, T, delta, speed, **kwargs):
    return cf_IDM_VS_equilibrium_jit(s0, v0, T, delta, speed)
