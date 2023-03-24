# -*- coding = uft-8 -*-
# @Time : 2022-04-06 12:04
# @Author : yzbyx
# @File : CFModel_OVM.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_OVM(CFModel):
    """
    a : 代表车辆动态的时间参数 (部分文章为1/tau)

    V0 : 优化速度系数

    m : 比例系数

    bf : V-S图中速度斜率极大值点(m) (原文包含前车长)

    bc : 期望最小净间距(m) (原文包含前车长)Gipps_a_main.py
    """
    PARAM = {
        'a': 2,             # 代表车辆动态的时间参数
        'V0': 16.8,         # 优化速度系数
        'm': 0.086,         # 比例系数
        'bf': 20,           # 加速度极大值点(m) (原文包含前车长，此处未包含)
        'bc': 2,            # 期望最小净间距(m) (原文包含前车长，此处未包含)
    }
    CFM_NAME = CFM.OPTIMAL_VELOCITY
    CFM_THESIS = {
        'Title': 'Analysis of optimal velocity model with explicit delay',
        'Author': 'Bando, M., Hasebe, K., Nakanishi, K., Nakayama, A.',
        'DOI': '10.1103/PhysRevE.58.5429',
        'Year': 1998,
    }

    def _calculate(self, interval, speed, acc, xOffset, length, leaderV, leaderA, leaderX, leaderL) -> dict:
        a = self.fParam['a']
        V0 = self.fParam['V0']
        m = self.fParam['m']
        bf = self.fParam['bf']
        bc = self.fParam['bc']

        bf += leaderL
        bc += leaderL       # 期望最小车头间距

        headway = leaderX - xOffset
        V_deltaX = V0 * (np.tanh(m * (headway - bf)) - np.tanh(m * (bc - bf)))
        # V_deltaX = V0 * (np.tanh(gap / b - C1) + C2)
        finalAcc = a * (V_deltaX - speed)

        finalV = speed + finalAcc * interval
        # if finalV < 0:
        #     finalAcc = - speed / interval
        #     finalV = 0
        xOffset += speed * interval + 0.5 * finalAcc * np.power(interval, 2)
        return {'xOffset': xOffset, 'speed': finalV, 'acc': finalAcc}
