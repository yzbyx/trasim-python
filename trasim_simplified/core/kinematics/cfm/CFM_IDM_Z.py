# -*- coding: utf-8 -*-
# @Time : 2024/1/19 21:05
# @Author : yzbyx
# @File : CFM_IDM_Z.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional
import numba
import numpy as np

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_IDM_Z(CFModel):
    """
    只包含IDM模型中的期望速度以及0速度差模块
    """
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        # -----模型属性------ #
        self.name = CFM.IDM
        self.thesis = 'Congested traffic states in empirical observations and microscopic simulations (2000)'

        # -----模型变量------ #
        self._d = f_param.get("d", 1.67)
        """舒适减速度"""

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
        return cf_IDM_Z_acc_jit(self._d, self.vehicle.v, self.vehicle.gap, self.vehicle.leader.v)

    def get_expect_dec(self):
        return self._d

    def get_expect_acc(self):
        return None

    def get_expect_speed(self):
        return None


@numba.njit()
def cf_IDM_Z_acc_jit(d, speed, gap, leaderV) -> dict:
    sStar = speed * (speed - leaderV) / (2 * np.sqrt(d))
    # sStar = s0 + np.max(np.array([0, s1 * np.sqrt(speed / v0) +
    #                               T * speed + speed * (speed - leaderV) / (2 * np.sqrt(omega * d))]))
    # 计算车辆下一时间步加速度
    return sStar / gap


def cf_IDM_Z_acc(d, speed, gap, leaderV, **kwargs) -> dict:
    return cf_IDM_Z_acc_jit(d, speed, gap, leaderV)


@numba.njit()
def cf_IDM_Z_equilibrium_jit():
    return None


def cf_IDM_Z_equilibrium(**kwargs):
    return None
