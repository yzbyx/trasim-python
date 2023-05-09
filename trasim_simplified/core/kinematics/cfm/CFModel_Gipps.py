# -*- coding = uft-8 -*-
# @Time : 2022-04-04 19:55
# @Author : yzbyx
# @File : CFModel_Gipps.py
# @Software : PyCharm
from typing import Optional, TYPE_CHECKING

import numpy as np

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle


class CFModel_Gipps(CFModel):
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        # -----模型属性------ #
        self.name = CFM.GIPPS
        self.thesis = 'A behavioural car-following model for computer simulation (1981)'

        # -----模型变量------ #
        self._a = f_param.get("a", 2)         # 最大期望加速度
        self._b = f_param.get("b", -3)       # 最大期望减速度
        self._v0 = f_param.get("v0", 20)      # 期望速度
        self._tau = f_param.get("tau", 0.7)     # 反应时间
        self._s = f_param.get("s", 6.5)       # 静止时正常最小车头间距（前车有效车长）
        self._b_hat = f_param.get("b_hat", -2.5)   # 预估前车最大期望减速度

    def _update_dynamic(self):
        if self.vehicle.lane.dt != self._tau:
            print(f"{self.name}模型的反应时间tau需要与仿真步长一致！")
            self.vehicle.lane.dt = self._tau

    def step(self, index):
        if self.vehicle.leader is None:
            return self.get_expect_acc()
        self._update_dynamic()
        f_param = [self._a, self._b, self._v0, self._tau, self._s, self._b_hat]
        return calculate(*f_param, self.vehicle.v, self.vehicle.x, self.vehicle.leader.v,
                         self.vehicle.x + self.vehicle.dhw)

    def equilibrium_state(self, speed, dhw, v_length):
        """
        通过平衡态速度计算三参数

        :param dhw: 平衡间距
        :param v_length: 车辆长度 [m]
        :param speed: 平衡态速度 [m/s]
        :return: KQV三参数的值 K[veh/km], Q[veh/h], V[km/h]
        """
        dhw = ((np.power(self._b * self._tau, 2) - np.power(speed - self._b * self._tau, 2)) / self._b +
               speed * self._tau + np.power(speed, 2) / self._b_hat) / 2 + self._s
        k = 1000 / dhw
        v = speed * 3.6
        q = k * v
        return {"K": k, "Q": q, "V": v}

    def get_expect_dec(self):
        return self._b

    def get_expect_acc(self):
        return self._a

    def get_expect_speed(self):
        return self._v0


def calculate(a, b, v0, tau, s, b_hat, speed, xOffset, leaderV, leaderX) -> dict:
    # 计算车头间距
    deltaX = leaderX - xOffset
    # 包络线公式限制
    vMax1 = speed + 2.5 * a * tau * (1 - speed / v0) * np.power(0.025 + speed / v0, 0.5)
    # 安全驾驶限制，注意此处的s为当前车与前车的期望车头间距
    vMax2 = b * tau + np.sqrt((b ** 2) * (tau ** 2) - b * (2 * (deltaX - s) - speed * tau - (leaderV ** 2) / b_hat))
    # 选取最小的速度限制作为下一时刻t+tau的速度
    vTau = np.min([vMax1, vMax2], axis=0)
    # 计算加速度和位置
    finalAcc = (vTau - speed) / tau
    return finalAcc
