# -*- coding = uft-8 -*-
# @Time : 2022-04-06 12:04
# @Author : yzbyx
# @File : CFModel_OVM.py
# @Software : PyCharm
from typing import Optional, TYPE_CHECKING

import numba
import numpy as np

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_OVM(CFModel):
    """
    a : 代表车辆动态的时间参数 (部分文章为1/tau)

    V0 : 优化速度系数

    m : 比例系数

    bf : V-S图中速度斜率极大值点(m) (原文包含前车长)

    bc : 期望最小净间距(m) (原文包含前车长)Gipps_a_main.py
    """
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        # -----模型属性------ #
        self.name = CFM.OPTIMAL_VELOCITY
        self.thesis = 'Analysis of optimal velocity model with explicit delay'

        # -----模型变量------ #
        self._a = f_param.get("a", 2)
        """代表车辆动态的时间参数"""
        self._V0 = f_param.get("V0", 16.8)
        """优化速度系数"""
        self._m = f_param.get("m", 0.086)
        """比例系数"""
        self._bf = f_param.get("bf", 20)
        """加速度极大值点(m) (原文包含前车长，此处未包含)"""
        self._bc = f_param.get("bc", 2)
        """期望最小净间距(m) (原文包含前车长，此处未包含)"""

    def _update_dynamic(self):
        pass

    def step(self, *args):
        if self.vehicle.leader is None:
            return self.get_expect_acc()
        self._update_dynamic()
        f_param = [self._a, self._V0, self._m, self._bf, self._bc]
        return calculate(*f_param, self.vehicle.v, self.vehicle.x,
                         self.vehicle.x + self.vehicle.dhw,
                         self.vehicle.leader.length)

    def equilibrium_state(self, speed, dhw, v_length):
        """
        通过平衡态速度计算三参数

        :param dhw: 平衡间距
        :param v_length: 车辆长度
        :param speed: 平衡态速度
        :return: KQV三参数的值
        """
        k = 1000 / dhw
        v = self._V0 * (np.tanh(self._m * (dhw - self._bf)) - np.tanh(self._m * (self._bc - self._bf))) * 3.6
        q = k * v
        return {"K": k, "Q": q, "V": v}

    def get_expect_dec(self):
        return 3.

    def get_expect_acc(self):
        return 3.

    def get_expect_speed(self):
        return min(self.get_speed_limit(), self._V0)


@numba.njit()
def calculate(a, V0, m, bf, bc, speed, xOffset, leaderX, leaderL):
    bf += leaderL
    bc += leaderL       # 期望最小车头间距

    headway = leaderX - xOffset
    V = V0 * (np.tanh(m * (headway - bf)) - np.tanh(m * (bc - bf)))
    # V = V0 * (np.tanh(gap / b - C1) + C2)
    finalAcc = a * (V - speed)

    return finalAcc
