# -*- coding = uft-8 -*-
# @Time : 2022-04-04 21:48
# @Author : yzbyx
# @File : CFModel_NonLinearGHR.py
# @Software : PyCharm

import numpy as np

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_NonLinearGHR(CFModel):
    """
    m : 速度系数

    l : 间距系数

    a : 与敏感度相关的系数

    tau : 反应时间
    """
    PARAM = {
        'm': 1,           # 速度系数
        'l': 1,           # 间距系数
        'a': 44.1 / 3.6,  # 与敏感度相关的系数
        'tau': 1.5          # 反应时间
    }
    CFM_NAME = CFM.NON_LINEAR_GHR
    CFM_THESIS = 'Nonlinear Follow-The-Leader Models of Traffic Flow (1961)'

    def _calculate(self, interval, speed, acc, xOffset, length, leaderV, leaderA, leaderX, leaderL) -> dict:
        m = self.fParam['m']
        l = self.fParam['l']
        a = self.fParam['a']
        tau = self.fParam['tau']
        # Newton-Raphson迭代法
        const1 = a * (leaderV - speed) / np.power(leaderX - xOffset, l)
        f = lambda vC: const1 * np.power(vC, m) - (vC - speed) / tau
        f_diff = lambda vC: const1 * m * np.power(vC, m - 1) - 1 / tau
        f_diff_diff = lambda vC: const1 * m * (m - 1) * np.power(vC, m - 2)
        g = lambda vC: f(vC) / f_diff(vC)
        g_diff = lambda vC: 1 - f(vC) * f_diff_diff(vC) / np.power(f_diff(vC), 2)   # 变多重根为单根
        resultPre = 50.0
        num = 0
        while num < 1e3:
            value1 = g(resultPre)
            value2 = g_diff(resultPre)
            result = resultPre - value1 / value2
            if np.isnan(result):
                break
            if np.abs(result - resultPre) < 1e-3:
                resultPre = result
                break
            resultPre = result
            num += 1
        # if np.abs(g(resultPre)) > 0.01:
        #     resultPre = np.nan
        # if resultPre < 0 and self.mode != RUNMODE.SILENT:
        #     warnings.warn(wm.SPEED_LESS_THAN_ZERO.format(self.driverID, resultPre), TrasimWarning)
        #     resultPre = np.nan
        #
        finalAcc = (resultPre - speed) / tau
        # if -8 < acc < 5:
        #     acc = np.nan
        xOffset += speed * tau + 0.5 * acc * np.power(tau, 2)
        return {'xOffset': xOffset, 'speed': resultPre, 'acc': finalAcc}


def test():
    clb = CFModel_NonLinearGHR(driverID='dummy')
    param = {
        'm': 0,  # 速度系数
        'l': 0.5,  # 间距系数
        'a': 2,  # 与敏感度相关的系数
        'tau': 1  # 反应时间
    }
    clb.updatefParam(param)
    testDict = {'interval': 0.1, 'speed': 20, 'acc': 0, 'xOffset': 0, 'length': 5,
                'leaderV': 10, 'leaderA': 0, 'leaderX': 100, 'leaderL': 5}
    testList = list(testDict.values())
    clb.step(*testList)


if __name__ == '__main__':
    test()
