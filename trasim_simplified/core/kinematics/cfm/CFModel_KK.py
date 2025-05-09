# -*- coding = uft-8 -*-
# @time : 2022-07-04 12:41
# @Author : yzbyx
# @File : CFModel_KK.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM, VehSurr

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle


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

    def __init__(self, f_param: dict[str, float]):
        super().__init__()
        # -----模型属性------ #
        self.name = CFM.KK
        self.thesis = 'Physics of automated driving in framework of three-phase traffic theory (2018)'

        self._v0 = f_param.get("v0", 33.3)

        # -----模型变量------ #
        self._s0 = f_param.get("s0", 2)
        """停车间距"""
        self._tau = f_param.get("tau", 1)
        """反应时间"""
        self._k = f_param.get("k", 3)
        """同步系数"""
        self._b = f_param.get("b", 1)
        """期望减速度"""
        self._a = f_param.get("a", 0.5)
        """舒适加速度（同舒适减速度）"""
        self._a_0 = 0.2 * self._a
        self._a_a = self._a_b = self._a

        self._p_a = f_param.get("p_a", 0.17)
        self._p_b = f_param.get("p_b", 0.1)
        self._p_0 = f_param.get("p_0", 0.005)
        self._p_1 = f_param.get("p_1", 0.3)

        self._v_01 = f_param.get("v_01", 10)
        self._v_21 = f_param.get("v_21", 15)

        self._v_safe_dispersed = f_param.get("v_safe_dispersed", False)
        """v_safe计算是否离散化时间"""

        self._delta_vr_2 = f_param.get("delta_vr_2", 5.)

        # self._time_wanted = f_param.get("time_wanted", 1.3)

        self.status = 0
        self.index = None
        self.ev: Optional["Vehicle"] = None

        self.scale = 1

    def _update_dynamic(self):
        self.scale = 1 if not self.veh_surr.ev.is_gaming else self.veh_surr.ev.game_factor

        vf_change = 0
        if self.veh_surr.ev.is_gaming:
            game_factor = self.veh_surr.ev.game_factor
            if game_factor < 1:
                game_factor = 1 / game_factor
                vf_change = (game_factor * 5) * np.sign(1 - self.veh_surr.ev.game_factor)

        self.dt = self.veh_surr.ev.dt
        self._vf = self.get_expect_speed() + vf_change

        self.v = self.veh_surr.ev.v
        self.l_length = self.veh_surr.cp.length
        self.gap = self.veh_surr.cp.x - self.veh_surr.ev.x - self.l_length - self._s0
        self.l_v = self.veh_surr.cp.v
        self.l_v_a = self.veh_surr.cp.a

        self.v_safe = cal_v_safe(
            self.v_safe_dispersed,
            self._tau * (self.scale if self.scale < 1 else 1),
            self.veh_surr.cp.v,
            self.gap,
            self.get_expect_dec(),
            self.get_expect_dec() * (self.scale if self.scale < 1 else 1)  # 前车期望减速度取当前车的期望减速度，下同
        )

    @property
    def v_safe_dispersed(self):
        return self._v_safe_dispersed

    @staticmethod
    def update_v_safe(cf_model):
        lane = cf_model.vehicle.lane
        has_v_safe = hasattr(lane, "_v_safe")
        if not has_v_safe or (has_v_safe and int(getattr(lane, "_update_step")) != lane.step_):
            v_safe = [cal_v_safe(
                cf_model.v_safe_dispersed,
                cf_model.dt,
                car.leader.v,
                car.gap,
                car.cf_model.get_expect_dec(),
                car.cf_model.get_expect_dec()  # 前车期望减速度取当前车的期望减速度，下同
            ) for car in lane.car_list[:-1]]
            if lane.is_circle:
                car = cf_model.vehicle.lane.car_list[-1]
                v_safe.append(cal_v_safe(
                    cf_model.v_safe_dispersed,
                    cf_model.dt,
                    car.leader.v,
                    car.gap,
                    car.cf_model.get_expect_dec(),
                    car.cf_model.get_expect_dec()
                ))

            v_a = [CFModel_KK.cal_v_a(
                cf_model.dt,
                car.gap,
                v_safe[i],
                car.v,
                car.cf_model.get_expect_acc(),
                car.is_first
            ) for i, car in enumerate(lane.car_list[:-1])]
            if not lane.is_circle:
                v_a.append(lane.car_list[-1].v)
            else:
                car = cf_model.vehicle.lane.car_list[-1]
                v_a.append(CFModel_KK.cal_v_a(
                    cf_model.dt,
                    car.gap,
                    v_safe[-1],
                    car.v,
                    car.cf_model.get_expect_acc(),
                    car.is_first
                ))

            setattr(lane, "_v_safe", v_safe)
            setattr(lane, "_v_a", v_a)
            setattr(lane, "_update_step", lane.step_)

        cf_model.v_safe = getattr(lane, "_v_safe")[cf_model.index]
        cf_model.v_a_list = getattr(lane, "_v_a")
        if lane.is_circle:
            cf_model.l_v_a = cf_model.v_a_list[cf_model.index + 1] \
                if (cf_model.index < len(cf_model.v_a_list) - 1) else cf_model.v_a_list[0]
        else:
            cf_model.l_v_a = cf_model.v_a_list[cf_model.index + 1] \
                if (cf_model.index < len(cf_model.v_a_list) - 1) else None
        return cf_model.l_v_a

    def step(self, veh_surr: VehSurr):
        self.veh_surr = veh_surr
        self.ev = veh_surr.ev
        if self.veh_surr.cp is None:
            speed = max(
                0., min(self.get_expect_speed(), self.veh_surr.ev.v + self._a * self.veh_surr.ev.dt)
            )
            return (speed - self.veh_surr.ev.v) / self.veh_surr.ev.dt
        self._update_dynamic()
        acc, self.status = self._calculate()
        return acc

    def _calculate(self):
        # ----an,bn计算---- #
        a_n, b_n = self.cal_an_bn()

        # ----G计算---- #
        self.G = cal_G(self._k, self._tau * self.scale, self._a, self.v, self.l_v)

        # ----v_c计算---- #
        v_c = self._cal_v_c(self.G, a_n, b_n)

        # if self.veh_surr.ev.is_gaming:
        #     game_factor = self.veh_surr.ev.game_factor
        #     if game_factor < 1:
        #         game_factor = 1 / game_factor
        #         v_c += (
        #                 (game_factor * 8 * self.dt) *
        #                 np.sign(1 - self.veh_surr.ev.game_factor)
        #         )

        # ----v_s计算---- #
        v_s = self._cal_v_s()

        # ----v_hat计算---- #
        v_hat = min(self._vf, v_s, v_c)
        # print("v_hat计算:", self._vf, v_s, v_c)

        # ----xi扰动计算---- #
        xi, S = self._cal_xi(v_hat, self.dt, self.v)

        # ----最终v和x计算---- #
        status = S
        final_speed = max(0., min(self._vf, v_hat + xi, self.v + self._a * self.dt, v_s))
        final_acc = (final_speed - self.v) / self.dt

        return final_acc, status

    def _cal_v_s(self):
        v_s = min(self.v_safe, self.gap / (self._tau * self.scale) + self.l_v_a)
        return v_s

    @staticmethod
    def v_hat_leader_on_ramp(_l: 'Vehicle', speed_limit_target, delta_vr_2):
        return max(0, min(30, (_l.v if _l is not None else speed_limit_target) + delta_vr_2))

    @staticmethod
    def cal_v_a(dt, gap, v_safe, v, expect_acc, is_first):
        if is_first:
            v_a = v
        else:
            v_a = max(0, min(v_safe, v, gap / dt) - expect_acc * dt)
        return v_a

    def get_expect_acc(self):
        return self._a

    def get_expect_dec(self):
        return self._b

    def get_expect_speed(self):
        return self._v0

    def get_speed_limit(self):
        return self.get_speed_limit()

    def get_max_dec(self):
        return 8

    def get_max_acc(self):
        return 8

    def get_time_wanted(self):
        return self._tau * self.scale * 1.3

    def get_time_safe(self):
        return self._tau * self.scale

    def get_com_acc(self):
        return self._a

    def get_com_dec(self):
        return self._b

    def get_safe_s0(self):
        return self._s0

    def cal_an_bn(self):
        r2 = self.random.random()
        P_0 = 1 if self.status == 1 else self.p_0_v(self.v)
        P_1 = self.p_2_v(self.v) if self.status == -1 else self._p_1
        # 随机加减速时间延迟
        a_n = self._a * self._sig_func(P_0 - r2)
        b_n = self._a * self._sig_func(P_1 - r2)
        return a_n, b_n

    def _cal_v_c(self, G, a_n, b_n):
        if not self.veh_surr.ev.is_gaming:
            if self.gap <= G:
                delta = max(-b_n * self.dt, min(a_n * self.dt, self.l_v - self.v))
                v_c = self.v + delta
            else:
                v_c = self.v + a_n * self.dt
        else:
            a_c = 0.3 * (self.gap - self.get_time_wanted() * self.v) + 0.3 * (self.l_v - self.v)
            v_c = self.v + a_c * self.dt
        return v_c

    def _cal_v_c_on_ramp(self, G, a_n, b_n, v_hat_leader, _l: 'Vehicle', gap):
        if gap <= G:
            delta_plus = max(- b_n * self._tau, min(a_n * self._tau, v_hat_leader - self.v))
            v_c = self.v + delta_plus
        else:
            v_c = self.v + a_n * self._tau
        return v_c

    def _cal_xi(self, v_hat, interval, speed) -> tuple[float, int]:
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
        """
        信号函数，小于0返回-1，大于等于0返回1
        """
        return 0 if x < 0 else 1

    def p_0_v(self, v):
        return 0.575 + 0.125 * min(1, v / self._v_01)

    def p_2_v(self, v):
        """

        :param v:
        :return:
        """
        return 0.48 + 0.32 * self._sig_func(v - self._v_21)


def cal_G(k_, tau_, a_, v, l_v):
    return max(0, k_ * tau_ * v + (1 / a_) * v * (v - l_v))


def cal_v_safe(v_safe_dispersed, tau, leaderV, gap, dec, leader_dec):
    """其中的dt为反应时间，同时也是离散化时间步长"""
    if v_safe_dispersed:
        alpha_l = int(leaderV / (leader_dec * tau))
        beta_l = leaderV / (leader_dec * tau) - alpha_l
        X_d_l = leader_dec * (tau ** 2) * (alpha_l * beta_l + 0.5 * alpha_l * (alpha_l - 1))
        alpha_safe = int(np.sqrt(2 * (X_d_l + gap) / (dec * (tau ** 2)) + 0.25) - 0.5)
        beta_safe = (X_d_l + gap) / ((alpha_safe + 1) * dec * (tau ** 2)) - alpha_safe / 2

        return dec * tau * (alpha_safe + beta_safe)
    else:
        x_d_l = (leaderV ** 2) / (2 * leader_dec)
        total_allow_dist = x_d_l + gap
        a = 1 / (2 * dec)
        v_safe = (- tau + np.sqrt(tau ** 2 + 4 * a * total_allow_dist)) / (2 * a)

        return v_safe


def picud(v_safe_dispersed, tau, v, l_v, gap, dec, leader_dec):
    if v_safe_dispersed:
        alpha = int(l_v / (dec * tau))  # 使用当前车的最大期望减速度
        beta = l_v / (dec * tau) - int(l_v / (dec * tau))
        xd_l = dec * tau * tau * (alpha * beta + 0.5 * alpha * (alpha - 1))

        alpha = int(v / (dec * tau))
        beta = v / (dec * tau) - int(v / (dec * tau))
        xd = dec * tau * tau * (alpha * beta + 0.5 * alpha * (alpha - 1))

        return (gap + xd_l) - (v * tau + xd)
    else:
        l_dec = leader_dec
        dec = dec
        xd_l = (l_v ** 2) / (2 * l_dec)
        xd = (v ** 2) / (2 * dec)
        return (gap + xd_l) - (v * tau + xd)


if __name__ == '__main__':
    print(cal_v_safe(True, 1, 30, 54 / 2 - 7.5, 3, 3))
    print(cal_v_safe(False, 1, 30, 54 / 2 - 7.5, 3, 3))
    print(picud(True, 1, 30, 30, 30, 3, 3))
    print(picud(False, 1, 30, 30, 54 - 7.5, 3, 3))
