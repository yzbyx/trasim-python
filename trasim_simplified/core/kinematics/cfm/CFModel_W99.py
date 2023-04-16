# -*- coding = uft-8 -*-
# @Time : 2022-04-04 10:55
# @Author : yzbyx
# @File : CFModel_W99.py
# @Software : PyCharm
############################
# The psycho-physical model of Wiedemann (10-Parameter version from 1999)
# references:
# code adapted from https://github.com/glgh/w99-demo
# (MIT License, Copyright (c) 2016 glgh)
############################
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_W99(CFModel):
    """
    'CC0': 1.50,    # 车辆静止时净间距(m)

    'CC1': 1.30,    # 期望车头时距(s)

    'CC2': 4.00,    # 车辆在有意加速前，除最小安全距离外所保持的间距(m)

    'CC3': -12.00,  # 车辆从注意到前方慢车到达到安全距离的时长(s)

    'CC4': -0.25,   # 无意识跟驰过程中相对速度的最小负值(m/s)

    'CC5': 0.35,    # 无意识跟驰过程中相对速度的最大正值(m/s)

    'CC6': 6.00,    # 振荡的速度依赖性(10^-4 rad/s)

    'CC7': 0.25,    # 无意识跟驰过程中车辆加速度(m/s^2)

    'CC8': 2.00,    # 起步时车辆的最大加速度(m/s^2)

    'CC9': 1.50,    # 80km时车辆最大加速度(m/s^2)

    'vDesire': 80 / 3.6,        # 期望速度(m/s)

    'aggressive': 0,            # 关于随机值的参数[0, 0.5]
    """

    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        # -----模型属性------ #
        self.name = CFM.WIEDEMANN_99
        self.thesis = 'Calibrating the Wiedemann’s vehicle-following model using mixed vehicle-pair interactions (2016)'

        # -----模型变量------ #
        self._CC0 = f_param.get("CC0", 1.50)
        """车辆静止时净间距(m)"""
        self._CC1 = f_param.get("CC1", 1.30)
        """期望车头时距(s)"""
        self._CC2 = f_param.get("CC2", 4.00)
        """车辆在有意加速前，除最小安全距离外所保持的间距(m)"""
        self._CC3 = f_param.get("CC3", -12.00)
        """车辆从注意到前方慢车到达到安全距离的时长(s)"""
        self._CC4 = f_param.get("CC4", -0.25)
        """无意识跟驰过程中相对速度的最小负值(m/s)"""
        self._CC5 = f_param.get("CC5", 0.35)
        """无意识跟驰过程中相对速度的最大正值(m/s)"""
        self._CC6 = f_param.get("CC6", 6.00)
        """振荡的速度依赖性(10^-4 rad/s)"""
        self._CC7 = f_param.get("CC7", 0.25)
        """无意识跟驰过程中车辆加速度(m/s^2)"""
        self._CC8 = f_param.get("CC8", 2.00)
        """起步时车辆的最大加速度(m/s^2)"""
        self._CC9 = f_param.get("CC9", 1.50)
        """80km时车辆最大加速度(m/s^2)"""
        self._vDesire = f_param.get("vDesire", 22.2)
        """期望速度(m/s) [22.2m/s->80km/h]"""
        self._aggressive = f_param.get("aggressive", 0)
        """关于随机值的参数[0, 0.5]"""

        self.status = None

    def _update_dynamic(self):
        pass

    def step(self, *args):
        f_param = [self._CC0, self._CC1, self._CC2, self._CC3, self._CC4, self._CC5, self._CC6, self._CC7,
                   self._CC8, self._CC9, self._vDesire, self._aggressive]
        if args:
            result = calculate(*f_param, *args)
        else:
            result = calculate(*f_param, self.status, self.vehicle.dynamic["speed"], self.vehicle.dynamic["xOffset"],
                             self.vehicle.leader.dynamic["speed"], self.vehicle.leader.dynamic["xOffset"],
                             self.vehicle.leader.static["length"])
            _, self.status = result
        return result

    def getThresholdValues(self, speed, gap, leaderV, leaderA):
        params = [self._CC0, self._CC1, self._CC2, self._CC3, self._CC4, self._CC5, self._CC6, self._aggressive]
        return getThresholdValues(*params, speed, gap, leaderV, leaderA)


def calculate(cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8, cc9, vDesire, aggressive, status,
              interval, speed, acc, xOffset, length, leaderV, leaderA, leaderX, leaderL):
    cc6 /= 10000

    dx = leaderX - xOffset - leaderL  # 车辆净间距
    dv = leaderV - speed  # 速度差

    sdxc, sdxo, sdxv, sdvc, sdvo = \
        getThresholdValues(cc0, cc1, cc2, cc3, cc4, cc5, cc6 * 10000, aggressive, speed, dx, leaderV, leaderA)

    finalAcc = acc

    if dv <= sdvo and dx <= sdxc:  # Decelerate - Increase Distance
        status = 'A'
        if speed > 0:
            if dv < 0:  # 减速区域的左下半部分/右下半部分
                if dx > cc0:  # 净间距还未到最小期望静止净间距
                    # 能够让下一时间步后车的速度与前车相同，且刚好不发生碰撞的最大减速度
                    finalAcc = min(leaderA + (dv ** 2) / (cc0 - dx), finalAcc)
                else:  # 净间距在最小期望静止净间距之内
                    # 下一时间步减速至前车加速度+到速度差与opdv差值的一半？
                    finalAcc = min(leaderA + 0.5 * (dv - sdvo), finalAcc)
                if finalAcc > - cc7:  # 强制减速值最大为无意识减速度
                    finalAcc = - cc7
                else:  # 其中的后半部分为最大减速度，受车辆此时速度的影响
                    finalAcc = max(finalAcc, -10 + 0.5 * np.sqrt(speed))
        else:
            finalAcc = 0
    elif dv < sdvc and dx < sdxv:  # Decelerate - Decrease Distance
        status = 'B'  # 后车的接近减速过程
        # 大意为采取的减速度为在到期望最小跟车间距时，速度与前车速度相同（假设前车加速度为0）的加速度的一半
        finalAcc = max(0.5 * (dv ** 2) / (sdxc - dx - 0.01), -10 + np.sqrt(speed))  # 此处最大减速度不如A状态的大
    elif dv < sdvo and dx < sdxo:  # Accelerate/Decelerate - Keep Distance
        status = 'f'  # 后车处于无意识跟驰状态
        if finalAcc <= 0:
            finalAcc = min(finalAcc, - cc7)  # 保持原加速度和无意识减速度的小值
        else:
            finalAcc = max(finalAcc, cc7)  # 保持原加速度和无意识加速度的大值
            if length >= 6.5:  # 考虑车型的加速度折减
                finalAcc *= 0.5
            # 同时加速过程需要判断是否超过期望速度，控制其不超过期望速度
            finalAcc = min(finalAcc, (vDesire - speed) / interval)
    else:  # Accelerate/Relax - Increase/Keep Speed
        status_temp = 'w'  # 自由状态
        if dx > sdxc:  # 大于最小跟驰距离
            if status == 'w':
                finalAcc = cc7  # 如果上一时间步也为自由状态，则加速度为无意识加速度值
            else:
                # 最大加速度为起步最大加速度+80km/h最大加速度*(速度与80km/h的小值)+随机值[0, 0.5]
                accMax = cc8 + cc9 * min(speed, 80 / 3.6) + myRandom(aggressive)  # 80km/h
                if dx < sdxo:
                    # 加速到sdxo，且与前车速度相同
                    finalAcc = min((dv ** 2) / (sdxo - dx), accMax)
                else:
                    finalAcc = accMax
            if length >= 6.5:  # 考虑车型的加速度折减
                finalAcc *= 0.5
            finalAcc = min(finalAcc, (vDesire - speed) / interval)
        else:
            finalAcc = 0
        status = status_temp
    # 自己添加的限制，下一时间步的速度不得小于0
    # finalAcc = max(finalAcc, (0 - speed) / interval)

    return finalAcc, status

def getThresholdValues(cc0, cc1, cc2, cc3, cc4, cc5, cc6, aggressive, speed, gap, leaderV, leaderA):
    cc6 = cc6 / 10000

    dx = gap  # 车辆净间距
    dv = leaderV - speed  # 速度差

    if leaderV <= 0:
        sdxc = cc0  # 前车速度小于等于0，则最小期望跟驰间距为最小停车净间距
    else:
        # 前车速度快或者减速度大时，选取当前车*期望车头时距，否则选取前车速度*期望车头时距
        v_slower = speed if (dv >= 0 or leaderA < -1) else leaderV + dv * (0.5 - myRandom(aggressive))
        sdxc = cc0 + cc1 * v_slower  # additional time headway
    sdxo = sdxc + cc2  # 最大无意识跟驰距离
    sdxv = sdxo + cc3 * (dv - cc4)  # perception threshold (near)

    sdv = cc6 * dx ** 2  # 在dX较小时该值几乎忽略不计，但刚进入跟驰范围时，大约为1的数量级
    sdvc = cc4 - sdv if leaderV > 0 else 0  # minimal closing dv
    sdvo = cc5 + sdv if speed > cc5 else sdv  # minimal opening dv

    return sdxc, sdxo, sdxv, sdvc, sdvo

def myRandom(seed):
    """根据一定规则，生成[0, 0.5]范围的随机数"""
    return seed
