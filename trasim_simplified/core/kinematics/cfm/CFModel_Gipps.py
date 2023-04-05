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
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float] ):
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
        self.dt = 0.1
        assert self.dt == self._tau, f"{self.name}模型的反应时间tau需要与仿真步长一致！"

    def step(self, *args):
        pass

    def calculate(*args):
        pass


    def _calculate(self, interval, speed, acc, xOffset, length, leaderV, leaderA, leaderX, leaderL) -> dict:
        a = self.fParam['a']
        b = self.fParam['b']
        v0 = self.fParam['v0']
        tau = self.fParam['tau']
        s = self.fParam['s']
        b_hat = self.fParam['b_hat']

        # 计算车头间距
        deltaX = leaderX - xOffset
        # 包络线公式限制
        vMax1 = speed + 2.5 * a * tau * (1 - speed / v0) * np.power(0.025 + speed / v0, 0.5)
        # 安全驾驶限制，注意此处的s为当前车与前车的期望车头间距
        vMax2 = b * tau + np.sqrt((b ** 2) * (tau ** 2) - b * (2 * (deltaX - s) - speed * tau - (leaderV ** 2) / b_hat))
        # 选取最小的速度限制作为下一时刻t+tau的速度
        self.status = 'free' if vMax1 < vMax2 else 'follow'
        vTau = min(vMax1, vMax2)
        # vTau = vMax2
        # 提示速度错误
        # if vTau < 0:
        #     if self.mode != RUNMODE.SILENT:
        #         message = f"vMax1: {vMax1}, vMax2: {vMax2}, current: {self.driverID}"
        #         warnings.warn(message, RuntimeWarning)
        #     vTau = 0  # 将负速度值设置为0
        # 计算加速度和位置
        finalAcc = (vTau - speed) / tau
        xOffset += speed * tau + 0.5 * finalAcc * np.power(tau, 2)
        return {'xOffset': xOffset, 'speed': vTau, 'acc': finalAcc}


if __name__ == '__main__':
    cfm = CFModel_Gipps('dummy')
    fParam = {
        'a': 1.9,  # 最大期望加速度
        'b': -3.1,  # 最大期望减速度
        'v0': 19.8,  # 期望速度
        'tau': 1.2,  # 反应时间
        's': 7.5,  # 静止时正常最小车头间距（前车有效车长）
        'b_hat': -2.5  # 预估前车最大期望减速度
    }
    cfm.updatefParam(fParam)
    cfm.step_only_return_acc(19.8, 30, 19.8)
