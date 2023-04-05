# -*- coding = uft-8 -*-
# @Time : 2022-04-04 14:22
# @Author : yzbyx
# @File : CFModel_IDM.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np
from numba import jit

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


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

    def _update_dynamic(self):
        pass

    def step(self, *args):
        """
        计算下一时间步的加速度

        :param args: 为了兼容矩阵计算设置的参数直接传递
        :return: 下一时间步的加速度
        """
        f_param = [self._s0, self._s1, self._v0, self._T, self._omega, self._d, self._delta]
        if args:
            return calculate(*f_param, *args)
        else:
            return calculate(*f_param, self.vehicle.dynamic["speed"], self.vehicle.dynamic["xOffset"],
                             self.vehicle.leader.dynamic["speed"], self.vehicle.leader.dynamic["xOffset"],
                             self.vehicle.leader.static["length"])

@jit(nopython=True)
def calculate(s0, s1, v0, T, omega, d, delta, speed, xOffset, leaderV, leaderX, leaderL) -> dict:
    sStar = s0 + s1 * np.sqrt(speed / v0) + T * speed + speed * (speed - leaderV) / (2 * np.sqrt(omega * d))
    # sStar = s0 + max(0, s1 * np.sqrt(speed / v0) + T * speed + speed * (speed - leaderV) / (2 * np.sqrt(omega * d)))
    # 计算与前车的净间距
    gap = leaderX - xOffset - leaderL
    # 计算车辆下一时间步加速度
    finalAcc = omega * (1 - np.power(speed / v0, delta) - np.power(sStar / gap, 2))

    return finalAcc
