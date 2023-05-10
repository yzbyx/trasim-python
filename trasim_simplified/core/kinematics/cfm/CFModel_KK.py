# -*- coding = uft-8 -*-
# @Time : 2022-07-04 12:41
# @Author : yzbyx
# @File : CFModel_KK.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle


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

    def __init__(self, car: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(car)
        # -----模型属性------ #
        self.name = CFM.KK
        self.thesis = 'Physics of automated driving in framework of three-phase traffic theory (2018)'

        # -----模型变量------ #
        self._d = f_param.get("d", 7.5)
        self._vf = f_param.get("vf", 30)

        self._k = f_param.get("k", 3)

        self._b = f_param.get("b", 1)
        self._a = f_param.get("a", 0.5)
        self._a_0 = 0.2 * self._a
        self._a_a = self._a_b = self._a

        self._p_a = f_param.get("p_a", 0.17)
        self._p_b = f_param.get("p_b", 0.1)
        self._p_0 = f_param.get("p_0", 0.005)
        self._p_1 = f_param.get("p_1", 0.3)

        self._v_01 = f_param.get("v_01", 10)
        self._v_21 = f_param.get("v_21", 15)

        self.status = 0
        self.index = None

    def _update_dynamic(self):
        self.dt = self.vehicle.lane.dt
        self._update_v_safe()

        self.v = self.vehicle.v
        self.gap = self.vehicle.gap
        self.l_v = self.vehicle.leader.v
        self.l_length = self.vehicle.leader.length

    def _update_v_safe(self):
        lane = self.vehicle.lane
        has_v_safe = hasattr(lane, "_v_safe")
        if not has_v_safe or (has_v_safe and int(getattr(lane, "_update_step") != lane.step_)):
            v_safe = [self.cal_v_safe(
                self.dt,
                car.leader.v,
                car.gap,
                car.cf_model.get_expect_dec(),
                car.leader.cf_model.get_expect_dec()
            ) for car in lane.car_list[:-1]]
            if lane.is_circle:
                car = self.vehicle.lane.car_list[-1]
                v_safe.append(self.cal_v_safe(
                    self.dt,
                    car.leader.v,
                    car.gap,
                    car.cf_model.get_expect_dec(),
                    car.leader.cf_model.get_expect_dec()
                ))

            v_a = [self._cal_v_a(
                self.dt,
                car.gap,
                v_safe[i],
                car.leader.v,
                car.cf_model.get_expect_acc()
            ) for i, car in enumerate(lane.car_list[:-1])]
            if not lane.is_circle:
                v_a.append(lane.car_list[-1].v)
            else:
                car = self.vehicle.lane.car_list[-1]
                v_a.append(self._cal_v_a(
                    self.dt,
                    car.gap,
                    v_safe[-1],
                    car.leader.v,
                    car.cf_model.get_expect_acc()
                ))

            setattr(lane, "_v_safe", v_safe)
            setattr(lane, "_v_a", v_a)
            setattr(lane, "_update_step", lane.step_)

        self.v_safe = getattr(lane, "_v_safe")[self.index]
        self.v_a = getattr(lane, "_v_a")
        self.l_v_a = self.v_a[self.index + 1] if (self.index <= len(self.v_a) - 2) else self.v_a[0]

    def step(self, index):
        self.index = index
        if self.vehicle.leader is None:
            return self.get_expect_acc()
        self._update_dynamic()
        acc, self.status = self._calculate()
        return acc

    def _calculate(self):
        # ----G计算---- #
        self.G = max(0, self._k * self.dt * self.v + 1 / self._a * self.v * (self.v - self.l_v))

        # ----v_s计算---- #
        v_s = self._cal_v_s()

        # ----v_c计算---- #
        v_c = self._cal_v_c(self.G)

        # ----v_hat计算---- #
        v_hat = min(self._vf, v_s, v_c)

        # ----xi扰动计算---- #
        xi, S = self._cal_xi(v_hat, self.dt, self.v)

        # ----最终v和x计算---- #
        status = S
        final_speed = max(0, min(self._vf, v_hat + xi, self.v + self._a * self.dt, v_s))
        final_acc = (final_speed - self.v) / self.dt

        return final_acc, status

    def _cal_v_s(self):
        v_s = min(self.v_safe, self.gap / self.dt + self.l_v_a)
        return v_s

    @staticmethod
    def cal_v_safe(dt, leaderV, gap, dec, leader_dec):
        alpha_l = int(leaderV / (leader_dec * dt))
        beta_l = leaderV / (leader_dec * dt) - alpha_l
        X_d_l = leader_dec * (dt ** 2) * (alpha_l * beta_l + 0.5 * alpha_l * (alpha_l - 1))
        if gap < 0: return 0  # TODO: 未能查出什么问题，目前遇到gap<0的情况，安全速度直接返回0
        alpha_safe = int(np.sqrt(2 * (X_d_l + gap) / (dec * (dt ** 2)) + 0.25) - 0.5)
        beta_safe = (X_d_l + gap) / ((alpha_safe + 1) * dec * (dt ** 2)) - alpha_safe / 2

        return dec * dt * (alpha_safe + beta_safe)

    @staticmethod
    def _cal_v_a(dt, gap, v_safe, l_v, expect_acc):
        return max(0, min(v_safe, l_v, gap / dt) - expect_acc * dt)

    def get_expect_acc(self):
        return self._a

    def get_expect_dec(self):
        return self._b

    def get_expect_speed(self):
        return self._vf

    def _cal_v_c(self, G):
        r2 = self.random.random()
        P_0 = 1 if self.status == 1 else self.p_0_v(self.v)
        P_1 = self.p_2_v(self.v) if self.status == -1 else self._p_1
        # 随机加减速时间延迟
        a_n = self._a * self._sig_func(P_0 - r2)
        b_n = self._a * self._sig_func(P_1 - r2)
        if self.gap <= G:
            delta = max(-b_n * self.dt, min(a_n * self.dt, self.l_v - self.v))
            v_c = self.v + delta
        else:
            v_c = self.v + a_n * self.dt
        return v_c

    def _cal_xi(self, v_hat, interval, speed):
        # ----xi扰动计算---- #
        r1 = self.random.random()
        xi_a = self._a_a * interval * self._sig_func(self._p_a - r1)
        xi_b = self._a_b * interval * self._sig_func(self._p_b - r1)
        if r1 < self._p_0:
            temp = -1
        elif self._p_0 <= r1 < 2 * self._p_0 and speed > 0:
            temp = 1
        else:
            temp = 0
        xi_0 = self._a_0 * interval * temp
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

    @staticmethod
    def _sig_func(x):
        return 0 if x < 0 else 1

    def p_0_v(self, v):
        return 0.575 + 0.125 * min(1, v / self._v_01)

    def p_2_v(self, v):
        return 0.48 + 0.32 * self._sig_func(v - self._v_21)
