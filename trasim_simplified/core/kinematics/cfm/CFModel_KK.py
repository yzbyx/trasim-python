# -*- coding = uft-8 -*-
# @Time : 2022-07-04 12:41
# @Author : yzbyx
# @File : CFModel_KK.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM, V_DYNAMIC, V_STATIC


class CFModel_KK(CFModel):
    """
    'd': 7.5m  # 最小停车间距

    'vf': 30m/s^2  # 最大速度

    'b': 1m/s^2  # 最大减速度

    'a': 0.5m/s^2  # 最大加速度

    'k': 3  # 系数

    'p_a': 0.17  # 概率

    'p_b': 0.1  # 概率

    'p_0': 0.005  # 概率

    'p_1' :0.3

    'v_01': 10

    'v_21': 15
    """
    PARAM = {
        'd': 7.5,
        'vf': 30,
        'b': 1,
        'a': 0.5,

        'k': 3,

        'p_a': 0.17,
        'p_b': 0.1,
        'p_0': 0.005,
        'p_1': 0.3,

        'v_01': 10,
        'v_21': 15,
    }
    CFM_NAME = CFM.KK
    CFM_THESIS = 'Physics of automated driving in framework of three-phase traffic theory (2018)'

    def __init__(self, driverID: str, pre_status=0):
        """
        :_param driverID: 车辆ID
        :_param pre_status: 上一时间步的车辆加减速状态，0为速度不变，-1为减速，1为加速
        """
        super().__init__(driverID)
        self.status = pre_status
        self.sig_func = lambda x: 0 if x < 0 else 1
        self.p_0_v = lambda v: 0.575 + 0.125 * min(1, v / self.v_01)
        self.p_2_v = lambda v: 0.48 + 0.32 * self.sig_func(v - self.v_21)
        self._freshParam()

    def _freshParam(self):
        self.d = self.fParam['d']
        self.vf = self.fParam['vf']

        self.k = self.fParam['k']

        self.b = self.fParam['b']
        self.a = self.fParam['a']
        self.a_0 = 0.2 * self.a
        self.a_a = self.a_b = self.a

        self.p_a = self.fParam['p_a']
        self.p_b = self.fParam['p_b']
        self.p_0 = self.fParam['p_0']
        self.p_1 = self.fParam['p_1']

        self.v_01 = self.fParam['v_01']
        self.v_21 = self.fParam['v_21']

    def updatefParam(self, param: dict) -> None:
        """
        更新跟驰参数

        :_param param: 包含待更新参数的字典
        """
        self.fParam.update(param)
        self._freshParam()

    def _calculate(self) -> dict:
        interval = self.driver.interval
        xOffset = self.driver.vehicle.dynamic[V_DYNAMIC.X_OFFSET]
        speed = self.driver.vehicle.dynamic[V_DYNAMIC.VELOCITY]
        leaderX = self.driver.leader.vehicle.dynamic[V_DYNAMIC.X_OFFSET]
        leaderV = self.driver.leader.vehicle.dynamic[V_DYNAMIC.VELOCITY]
        d_l = self.driver.leader.fRule.d

        # 车头间距-最小停车间距
        if leaderX < xOffset:
            leaderX += self.driver.maxOffset
        g = leaderX - xOffset - d_l
        # G计算
        G = max(0, self.k * interval * speed + 1 / self.a * speed * (speed - leaderV))
        # ----v_s计算---- #
        # v_s = self._cal_v_s_new(g, interval)
        v_s = self._cal_v_s(g, interval, leaderV)

        # ----v_c计算---- #
        v_c = self._cal_v_c(g, G, interval, speed, leaderV)

        # ----v_hat计算---- #
        v_hat = min(self.vf, v_s, v_c)

        # ----xi扰动计算---- #
        xi, S = self._cal_xi(v_hat, interval, speed)

        # ----最终v和x计算---- #
        self.status = S
        final_speed = max(0, min(self.vf, v_hat + xi, speed + self.a * interval, v_s))
        final_acc = (final_speed - speed) / interval
        final_xOffset = xOffset + final_speed * interval + 0.5 * final_acc * (interval ** 2)

        return {'xOffset': final_xOffset, 'speed': final_speed, 'acc': final_acc}

    def _cal_v_s(self, g, interval, leaderV):
        b_l = self.driver.leader.fRule.b
        v_safe = self.b * (-interval + np.sqrt(interval ** 2 +
                                               2 / self.b * (g + leaderV ** 2 / (2 * b_l))))
        v_l_a = self._cal_v_l_a()
        v_s = min(v_safe, g / interval + v_l_a)
        return v_s

    def _cal_v_s_new(self, g, interval):
        v_safe = self.cal_v_safe()[0]

        fRule_l: CFModel_KK = self.driver.leader.fRule

        v_safe_l, speed_l, g_l, interval_l = fRule_l.cal_v_safe()

        v_l_a = max(0, min(v_safe_l, speed_l, g_l / interval_l) - self.a * interval)

        v_s = min(v_safe, g / interval + v_l_a)

        return v_s

    def cal_v_safe(self):
        interval = self.driver.interval
        speed = self.driver.vehicle.dynamic[V_DYNAMIC.VELOCITY]
        xOffset = self.driver.vehicle.dynamic[V_DYNAMIC.X_OFFSET]
        leaderX = self.driver.leader.vehicle.dynamic[V_DYNAMIC.X_OFFSET]
        leaderV = self.driver.leader.vehicle.dynamic[V_DYNAMIC.VELOCITY]
        d_l = self.driver.leader.fRule.d

        if leaderX < xOffset:
            leaderX += self.driver.maxOffset
        g = leaderX - xOffset - d_l

        leader = self.driver.leader
        leader_fRule: CFModel_KK = leader.fRule
        leader_fParam = leader_fRule.fParam
        b_l = leader_fParam['b']

        alpha_l = leaderV // (b_l * interval)
        beta_l = leaderV / (b_l * interval) - alpha_l
        X_d_l = b_l * (interval ** 2) * (alpha_l * beta_l + 0.5 * alpha_l * (alpha_l - 1))
        alpha_safe = int(np.sqrt(2 * (X_d_l + g) / (self.b * interval ** 2) + 0.25) - 0.5)
        beta_safe = (X_d_l + g) / ((alpha_safe + 1) * self.b * interval ** 2) - alpha_safe / 2
        v_safe = self.b * interval * (alpha_safe + beta_safe)

        return v_safe, speed, g, interval

    def _cal_v_l_a(self):
        leader = self.driver.leader
        leader_fRule: CFModel_KK = leader.fRule
        interval = leader.interval
        b_l = leader_fRule.b
        b_l_l = leader.leader.fRule.b
        d_l_l = leader.leader.vehicle.static[V_STATIC.LENGTH]
        leader_leader_X = leader.leader.getDynamic(V_DYNAMIC.X_OFFSET)
        leader_X = leader.getDynamic(V_DYNAMIC.X_OFFSET)
        if leader_leader_X < leader_X:
            leader_leader_X += self.driver.maxOffset
        g_l = leader_leader_X - leader_X - d_l_l
        leaderV_l = leader.leader.getDynamic(V_DYNAMIC.VELOCITY)

        try:
            v_safe_l = b_l * (-interval + np.sqrt(interval ** 2 +
                                              2 / b_l * (g_l + leaderV_l ** 2 / (2 * b_l_l))))
        except RuntimeWarning as w:
            print(f'b_l: {b_l}, g_l: {g_l}, v_l: {leaderV_l}')

        v_l_a = max(0, min(v_safe_l, leaderV_l, g_l / interval) - self.a * self.driver.interval)
        return v_l_a

    def _cal_v_c(self, g, G, interval, speed, leaderV):
        r2 = self.random.random()
        P_0 = 1 if self.status == 1 else self.p_0_v(speed)
        P_1 = self.p_2_v(speed) if self.status == -1 else self.p_1
        # 随机加减速时间延迟
        a_n = self.a * self.sig_func(P_0 - r2)
        b_n = self.a * self.sig_func(P_1 - r2)
        if g <= G:
            delta = max(-b_n * interval, min(a_n * interval, leaderV - speed))
            v_c = speed + delta
        else:
            v_c = speed + a_n * interval
        return v_c

    def _cal_xi(self, v_hat, interval, speed):
        # ----xi扰动计算---- #
        r1 = self.random.random()
        xi_a = self.a_a * interval * self.sig_func(self.p_a - r1)
        xi_b = self.a_b * interval * self.sig_func(self.p_b - r1)
        if r1 < self.p_0:
            temp = -1
        elif self.p_0 <= r1 < 2 * self.p_0 and speed > 0:
            temp = 1
        else:
            temp = 0
        xi_0 = self.a_0 * interval * temp
        # 施加随机超额速度扰动xi_a或xi_b，在加速度为0时施加随机速度扰动xi_0
        if v_hat < speed:
            S = -1
            xi = - xi_b
        elif v_hat > speed:
            S = 1
            xi = xi_a
        else:
            S = 0
            xi = xi_0
        return xi, S
