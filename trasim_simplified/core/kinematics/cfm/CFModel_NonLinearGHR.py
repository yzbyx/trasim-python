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
        pass
