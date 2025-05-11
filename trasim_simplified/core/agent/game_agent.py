# -*- coding: utf-8 -*-
# @time : 2025/3/22 22:13
# @Author : yzbyx
# @File : game_agent.py
# Software: PyCharm
import itertools
from typing import TYPE_CHECKING, Iterable, Optional

import cvxpy
import numpy as np
import pandas as pd
import sympy as sp
from sympy import Interval

from trasim_simplified.core.agent import Vehicle
from trasim_simplified.core.agent.base_agent import Base_Agent
from trasim_simplified.core.agent.mpc_solver import MPC_Solver
from trasim_simplified.core.agent.ref_path import ReferencePath
from trasim_simplified.core.agent.utils import get_xy_quintic, get_y_guess, interval_intersection
from trasim_simplified.core.constant import VehSurr, GameRes, TrajData, V_TYPE, TrajPoint

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


def cal_other_cost(veh, cal_cost, rho_hat, traj, other_traj_s,
                   v_length_s, route_cost=0, print_cost=False):
    if cal_cost:
        cost_lambda = veh.cal_cost_by_traj(
            traj, other_traj_s, v_length_s,
            return_lambda=True, route_cost=route_cost,
            print_cost=print_cost
        )
        cost_hat = cost_lambda(rho_hat) if cal_cost else np.nan
        real_cost = cost_lambda(veh.rho)
    else:
        cost_hat = np.nan
        real_cost = np.nan
        cost_lambda = None
    return cost_hat, real_cost, cost_lambda


class Game_Vehicle(Base_Agent):
    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.SCALE = 0.4
        self.DELTA_V = 10

    def get_rho_hat_s(self, vehicles):
        rho_hat_s = []
        for v in vehicles:
            if v is not None:
                if v.ID not in self.rho_hat_s:
                    self.rho_hat_s[v.ID] = [0, 1]
                rho_hat_s.append(np.mean(self.rho_hat_s[v.ID]))
            else:
                rho_hat_s.append(None)
        return rho_hat_s

    def set_game_stra(self, stra, leader):
        """设置策略"""
        self.is_gaming = True
        self.game_factor = stra
        self.game_leader = leader

    def clear_game_stra(self):
        """清除策略"""
        self.is_gaming = False
        self.game_factor = None
        self.game_leader = None

    def cal_safe_cost(self, traj, other_traj, v_length, current_gap):
        """计算与其他轨迹的安全成本"""
        if other_traj is None:
            return -np.inf
        assert len(traj) == len(other_traj), f"The length of traj is not equal to the other_traj."
        # 获取同一车道的轨迹，简化为如果横向y距离小于车身宽度，则认为在同一车道
        lane_indexes = self.get_lane_indexes(traj[:, 3])
        other_lane_indexes = self.get_lane_indexes(other_traj[:, 3])
        indexes = np.where((lane_indexes == other_lane_indexes))[0]
        traj = traj[indexes]
        other_traj = other_traj[indexes]

        # 最大cost求解
        if len(traj) == 0:
            return -np.inf

        # 计算安全成本
        safe_cost_list = []
        for point1, point2 in zip(traj, other_traj):
            dhw = point2[0] - point1[0]
            if dhw >= 0:
                if dhw <= v_length + point1[1] * self.time_safe:
                    # 车辆间距大于安全距离
                    safe_cost_list.append(1)
                    continue
                gap = dhw - v_length
            else:  # 自车在前
                if abs(dhw) <= self.length + point2[1] * self.time_safe:
                    # 车辆间距小于安全距离
                    safe_cost_list.append(1)
                    continue
                gap = abs(dhw) - self.length
            safe_cost_list.append(
                (current_gap - gap) / current_gap  # 计算安全成本
            )

        safe_cost = max(safe_cost_list)
        return safe_cost

    def cal_cost_by_traj(self, traj, other_traj_s, v_length_s,
                         rho=None, return_lambda=False,
                         route_cost=0, return_sub_cost=False, print_cost=False):
        """
        :param print_cost:
        :param traj: 预测轨迹 x, dx, ddx, y, dy, ddy
        :param other_traj_s: 其他车辆的轨迹
        :param v_length_s: 其他车辆的长度
        :param rho: 激进系数
        :param route_cost: 路径成本
        :param return_sub_cost: 是否返回子成本
        :param return_lambda: 是否返回lambda函数
        """
        # 计算安全成本
        safe_cost_list = []
        current_gap = self.v * self.time_wanted
        for other_traj, v_length in zip(other_traj_s, v_length_s):
            if other_traj is None:
                continue
            safe_cost_list.append(self.cal_safe_cost(traj, other_traj, v_length, current_gap))

        if len(safe_cost_list) == 0:
            safe_cost = 0
        else:
            safe_cost = self.k_s * max(max(safe_cost_list), -1)
            # safe_cost = self.k_s * max(safe_cost_list)
        # 舒适性计算
        jerk_x_max = max(abs(np.diff(traj[:, 2])))
        jerk_y_max = max(abs(np.diff(traj[:, 5])))

        com_cost = self.k_c * (0.5 * jerk_x_max + 0.5 * jerk_y_max) / self.JERK_MAX  # 舒适性计算
        # 效率计算 单位时间内的平均速度-初始速度
        vx = traj[:, 1]
        eff_cost = self.k_e * (vx[0] - np.mean(vx)) / self.DELTA_V  # 效率计算

        route_cost = route_cost  # 路径成本

        if print_cost:
            print("traj_cost:", safe_cost, com_cost, eff_cost, route_cost)

        def cost(rho_, is_print=False):
            total_cost = (
                    (1 - rho_) * (safe_cost + com_cost + route_cost) +
                    rho_ * eff_cost
            )
            if is_print:
                print("call lambda", rho_, safe_cost, com_cost, eff_cost, route_cost)
            return total_cost

        if return_lambda:
            return cost

        if return_sub_cost:
            return safe_cost, com_cost, eff_cost, route_cost

        return cost(rho)

    def get_strategies(self, is_lc: bool = False, single_stra=False, lc_after=False):
        if single_stra:
            strategy = [1]
        else:
            if is_lc:
                if lc_after:
                    strategy = itertools.product(np.arange(1, 5.1, 2), [0])
                else:
                    strategy = itertools.product(np.arange(1, 10.1, 2), (1, 0))
            else:
                # strategy = [1 / 3, 1 / 2.5, 1 / 2, 1 / 1.5, 1, 1.5, 2, 2.5, 3]
                strategy = [0.6, 0.8, 1, 1.2, 1.4]
        return strategy

    def _make_dummy_agent(self, lane, type_, id_, length, x, y):
        dummy_agent = Game_O_Vehicle(lane, type_, id_, length)
        dummy_agent.x = x
        dummy_agent.y = y
        dummy_agent.speed = self.speed
        dummy_agent.cf_model = self.cf_model
        return dummy_agent


class Game_A_Vehicle(Game_Vehicle):
    """自动驾驶车辆"""
    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.N_MPC = 5
        self.mpc_solver: Optional[MPC_Solver] = None

        self.is_game_leader = False
        self.last_cal_step = -1
        self.state = None

        self.re_cal_step = 10

    def lc_intention_judge(self):
        if self.opti_game_res is not None and self.target_lane == self.lane:
            self.lc_direction = 0

        if (self.opti_game_res is not None
                and abs(self.y_c - self.target_lane.y_center) < 0.1
                and self.v_lat < 0.1 and abs(self.yaw) < 1 / 180 * np.pi):
            # 恢复TR、TF的TIME_WANTED
            TR = self.opti_game_res.TR
            TR.clear_game_stra()
            TF = self.opti_game_res.TF
            TF.clear_game_stra()

            self.is_gaming = False
            self.is_game_leader = False

            self.lane_changing = False
            self.lc_conti_time = 0
            self.lc_direction = 0
            self.opti_game_res = None
            self.opti_gap = None
            self.game_res_list = None

        if not self.lane_changing:
            super().lc_intention_judge()

    def lc_decision_making(self, **kwargs):
        """判断是否换道"""
        super().lc_decision_making(set_lane_changing=False)
        if self.no_lc:
            return
        if self.is_gaming and self.is_game_leader and self.last_cal_step == 1 and self.lane is not None:
            self.update_rho_hat()
        self.state = self.get_state_for_traj()
        if self.lane == self.target_lane and self.lane_changing:
            self.lc_direction = 0
        if self.lane_changing or self.lc_direction != 0:
            # LC轨迹优化，如果博弈选择不换道，需要重新更新self.lane_changing，is_gaming以及target_lane
            if self.lane_changing:
                self.lc_conti_time += self.lane.dt
            if (
                    (self.lane_changing and self.last_cal_step >= self.re_cal_step)
                    or self.risk_2d < self.ttc_star
                    or (not self.lane_changing and self.lc_direction != 0)
            ):
                self.stackel_berg()

            self.last_cal_step += 1

    def cal_vehicle_control(self):
        """横向控制"""
        # if self.lane_changing is False or self.lane == self.target_lane:
        if self.lane_changing is False:
            delta = self.cf_lateral_control()
            acc = self.cf_model.step(self.pack_veh_surr())
            next_acc_block = np.inf
            if self.lane.index not in self.destination_lane_indexes:
                next_acc_block = self.cf_model.step(VehSurr(ev=self, cp=self.lane.road.end_weaving_block_veh))
            # print("acc", acc, "next_acc_block", next_acc_block)
            acc = min(acc, next_acc_block)
        else:
            # 横向控制
            acc, delta, is_end = self.mpc_solver.step_mpc()  # 默认计算出的符合约束的加速度和转向角
            if is_end:
                self.lane_changing = False
                self.lc_conti_time = 0
                print("MPC end")

        self.next_acc = acc
        self.next_delta = delta

    def reset_lc_game(self, remove_lc_direction=True):
        """重置博弈状态"""
        if self.opti_game_res is not None:
            self.opti_game_res.TR.clear_game_stra()
            if isinstance(self.opti_game_res.TF, Game_A_Vehicle):
                self.opti_game_res.TF.clear_game_stra()
        self.is_gaming = False
        self.game_factor = 1
        self.lane_changing = False
        if remove_lc_direction:
            self.lc_direction = 0
        self.target_lane = self.lane
        self.opti_game_res = None
        self.mpc_solver = None
        self.lc_conti_time = 0
        self.last_cal_step = 0
        self.game_leader = None

    def update_rho_hat(self):
        """求解rho的范围，入股TR的rho_hat不在范围内，更新TR的rho_hat"""
        TR = self.opti_game_res.TR
        if isinstance(TR, Game_H_Vehicle):
            # 第一次估计TR的rho_hat默认为0.5
            # rho_hat = self._get_rho_hat_s([TR])[0]
            TR_esti_lambda = self.opti_game_res.TR_esti_lambda
            TR_real_lambda = self.opti_game_res.TR_real_lambda

            # 构建不等式求解问题
            rho_real = sp.symbols("rho_real")
            # 定义不等式
            inequality = sp.Le(TR_real_lambda(rho_real), TR_esti_lambda(rho_real))  # f(x) <= g(x)
            # print("inequality: ", inequality)
            # 求解不等式
            solution = sp.solve_univariate_inequality(
                inequality, rho_real, relational=False, domain=Interval(0, 1))

            # 获取区间的上下限
            try:
                lower_bound = float(solution.start)
                upper_bound = float(solution.end)
            except:
                lower_bound = 0
                upper_bound = 1
            print("cal rho_hat", lower_bound, upper_bound)
            # 更新TR的rho_hat
            rho_hat_range_before = self.rho_hat_s[TR.ID]
            # 取交集
            rho_hat_range = interval_intersection(
                rho_hat_range_before,
                (lower_bound, upper_bound), print_flag=True
            )
            if rho_hat_range is not None:
                self.rho_hat_s[TR.ID] = rho_hat_range

            # print(
            #     "solution: ", solution,
            #     "ori rho_hat", self.rho_hat_s[TR.ID],
            #     "new rho_hat", self.rho_hat_s[TR.ID]
            # )

    def stackel_berg(self):
        """基于stackelberg主从博弈理论的换道策略，计算得到参考轨迹（无论是否换道）"""
        # 计算不同策略（换道时间）的最优轨迹和对应效用值
        # cal_game_matrix需要更新周边车辆的策略，stackel_berg函数只更新自身的策略
        self.game_res_list = self.cal_game_matrix()
        game_cost_list = [res.EV_cost for res in self.game_res_list]
        if len(game_cost_list) == 0:
            self.reset_lc_game()
            print("没有可行的博弈策略")
            return
        # 获取最优策略
        min_cost_idx = np.argmin(game_cost_list)
        opti_game_res = self.game_res_list[min_cost_idx]
        if opti_game_res.EV_cost == np.inf:
            self.reset_lc_game()
            print("没有可行的博弈策略")
            return

        # 获取最优策略对应的轨迹
        opti_df = opti_game_res.EV_opti_series

        # a0~a5, b0~b5, step, y1, ego_stra, TF_stra, TR_stra, PC_stra, ego_cost, TF_cost, TR_cost, PC_cost
        # 选择效用值最小的策略
        x_opt = opti_df.iloc[: 12].to_numpy().astype(float)
        T_opt, is_lc = opti_df.iloc[14]

        if (is_lc == 0 and abs(self.y_c - self.lane.y_center) < 0.5
                and abs(self.yaw) < 1 / 180 * np.pi):
            self.reset_lc_game()
            return

        self.reset_lc_game(remove_lc_direction=False)

        self.opti_game_res: GameRes = opti_game_res
        TR_stra = self.opti_game_res.TR_stra
        self.opti_game_res.TR.set_game_stra(TR_stra, self)
        if isinstance(self.opti_game_res.TF, Game_A_Vehicle):
            self.opti_game_res.TF.set_game_stra(self.opti_game_res.TF_stra, self)
        self.opti_game_res.EV_lc_step = round(T_opt / self.dt)
        self.opti_game_res.EV_opti_traj = self.cal_ref_path(x_opt)

        self.lane_changing = True
        self.is_gaming = True
        self.game_factor = 1
        if self.lc_direction == -1:
            self.target_lane = self.left_lane
        elif self.lc_direction == 1:
            self.target_lane = self.right_lane
        else:
            self.target_lane = self.lane
        self.is_game_leader = True

    def get_y_constraint(self, lc_direction):
        if lc_direction == -1:
            y1 = self.lane.left_neighbour_lane.y_center
            if self.y < self.lane.y_center:
                y_limit_low = self.lane.y_right
            else:
                y_limit_low = self.lane.y_center - self.lane.width / 4
            target_lane = self.lane.left_neighbour_lane
            y_limit_up = target_lane.y_center + target_lane.width / 4
            y_middle = target_lane.y_right
        elif lc_direction == 1:
            y1 = self.lane.right_neighbour_lane.y_center
            if self.y < self.lane.y_center:
                y_limit_up = self.lane.y_center + self.lane.width / 4
            else:
                y_limit_up = self.lane.y_left
            target_lane = self.lane.right_neighbour_lane
            y_limit_low = target_lane.y_center - target_lane.width / 4
            y_middle = target_lane.y_left
        else:
            y1 = self.lane.y_center
            if self.y < y1 - self.lane.width / 4:
                y_limit_low = self.lane.y_right
                y_limit_up = self.lane.y_center + self.lane.width / 4
            elif self.y > y1 + self.lane.width / 4:
                y_limit_low = self.lane.y_center - self.lane.width / 4
                y_limit_up = self.lane.y_left
            else:
                y_limit_low = self.lane.y_center - self.lane.width / 4
                y_limit_up = self.lane.y_center + self.lane.width / 4
            y_middle = self.lane.y_center

        return y1, y_limit_low, y_limit_up, y_middle

    def cal_game_matrix(self):
        """backward induction method
        :return: opti_df (a0~a5, b0~b5, ego_stra, TF_stra, TR_stra, PC_stra,
         ego_cost, TF_cost, TR_cost, PC_cost)
        """
        game_res_list = []
        single_stra = False
        for gap in [-1, 0, 1]:
            TR, TF, PC, CR = self._no_car_correction(gap, self.lc_direction)
            TR_real_stra, TF_stra, EV_stra, CP_stra, CR_stra = np.nan, np.nan, np.nan, np.nan, np.nan
            TR_real_cost, TF_cost, EV_cost, CP_cost, CR_cost = np.nan, np.nan, np.nan, np.nan, np.nan
            TR_esti_lambda = TR_real_lambda = None
            TR_real_EV_stra = np.nan
            if isinstance(TR, Game_H_Vehicle) and isinstance(TF, Game_H_Vehicle):
                # ego与TR的博弈
                TR_strategy = TR.get_strategies(is_lc=False, single_stra=single_stra)
                ego_strategy = self.get_strategies(
                    is_lc=self.lc_direction != 0 or self.lane_changing,  # 意图或者已经换道
                    lc_after=self.lc_direction == 0  # 是否过线
                )

                # ego_stra, TF_stra, TR_stra, CR_stra, CP_stra
                cost_df, traj_data = self._cal_cost_df(
                    stra_product=itertools.product(ego_strategy, [None], TR_strategy, [None], [None]),
                    cal_TF_cost=False, cal_TR_cost=True, cal_CP_cost=False, cal_CR_cost=False,
                    TF=TF, TR=TR, PC=PC, CR=CR
                )

                # 根据估计的TR成本函数计算最小的当前车辆cost
                # ego_stra包含了换道时间和换道方向，提取换道行为的cost_df_temp
                # cost_df_temp = cost_df[
                #     (cost_df["ego_cost"] != np.inf) & (cost_df["ego_stra"].apply(lambda x: x[1]))]
                # cost_df_temp = cost_df[cost_df["ego_cost"] != np.inf]
                cost_df_temp = cost_df
                if len(cost_df_temp) == 0:
                    continue
                min_TR_cost_idx = cost_df_temp.groupby('ego_stra')['TR_cost_hat'].idxmin()  # 找到最小的TR_cost值
                min_cost_idx = cost_df_temp.loc[min_TR_cost_idx]["ego_cost"].idxmin()  # 找到最小的cost值
                EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
                EV_opti_series = cost_df_temp.loc[min_cost_idx]
                index = EV_opti_series["index"]
                traj_data_opti = traj_data[index]

                min_TR_cost_idx_real = cost_df_temp.groupby('ego_stra')['TR_cost_real'].idxmin()  # 找到最小的TR_cost值
                min_cost_idx_real = cost_df_temp.loc[min_TR_cost_idx_real]["ego_cost"].idxmin()  # 找到最小的cost值
                TR_real_stra = cost_df_temp.loc[min_cost_idx_real]["TR_stra"]
                TR_real_cost = cost_df_temp.loc[min_cost_idx_real]["TR_cost_real"]
                TR_real_EV_stra = cost_df_temp.loc[min_cost_idx_real]["ego_stra"]

                TR_index = cost_df_temp[
                    (cost_df_temp["ego_stra"] == EV_stra) & (cost_df_temp["TR_stra"] == TR_real_stra)
                    ]["index"].values[0]
                traj_data_real = traj_data[TR_index]
                TR_esti_lambda = traj_data_opti.TR_cost_lambda
                TR_real_lambda = traj_data_real.TR_cost_lambda

                # u_hat = TR_esti_lambda(TR.rho)
                # u_real = TR_real_lambda(TR.rho)
                # print(
                #     "TR_cost_hat: ", u_hat,
                #     "TR_cost_real: ", u_real
                # )
                # assert u_hat >= u_real

                # print(
                #     "TR_cost_hat: ", EV_opti_series["TR_cost_hat"],
                #     "TR_real_cost: ", TR_real_cost
                # )

                # print(cost_df[["ego_stra", "ego_cost"]])
            elif isinstance(TR, Game_H_Vehicle) and isinstance(TF, Game_A_Vehicle):
                # ego与TR的博弈，ego与TF的合作
                TR_strategy = TR.get_strategies(is_lc=False, single_stra=single_stra)
                TF_strategy = TF.get_strategies(is_lc=False, single_stra=single_stra)
                ego_strategy = self.get_strategies(
                    is_lc=self.lc_direction != 0 or self.lane_changing,  # 意图或者已经换道
                    lc_after=self.lc_direction == 0  # 是否过线
                )

                cost_df, traj_data = self._cal_cost_df(
                    stra_product=itertools.product(ego_strategy, TF_strategy, TR_strategy, [None], [None]),
                    cal_TF_cost=True, cal_TR_cost=True, TF=TF, TR=TR, PC=PC, CR=CR
                )

                cost_df["total_cost"] = cost_df["ego_cost"] + TF.game_co * cost_df["TF_cost_real"]  # 合作效用
                cost_df_temp = cost_df
                # cost_df_temp = cost_df[cost_df["total_cost"] != np.inf]
                if len(cost_df_temp) == 0:
                    continue
                # 找到最小的TR_cost值
                min_TR_cost_idx = cost_df_temp.groupby(by=["ego_stra", "TF_stra"])['TR_cost_hat'].idxmin()
                min_cost_idx = cost_df_temp.loc[min_TR_cost_idx]["total_cost"].idxmin()  # 找到最小的cost值
                TF_stra = cost_df_temp.loc[min_cost_idx]["TF_stra"]
                EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                TF_cost = cost_df_temp.loc[min_cost_idx]["TF_cost_real"]
                EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
                EV_opti_series = cost_df_temp.loc[min_cost_idx]
                index = EV_opti_series["index"]
                traj_data_opti = traj_data[index]

                # 更新TR的策略
                min_TR_cost_idx_real = cost_df_temp.groupby(by=["ego_stra", "TF_stra"])['TR_cost_real'].idxmin()
                min_cost_idx_real = cost_df_temp.loc[min_TR_cost_idx_real]["total_cost"].idxmin()
                TR_real_stra = cost_df_temp.loc[min_cost_idx_real]["TR_stra"]
                TR_real_cost = cost_df_temp.loc[min_cost_idx_real]["TR_cost_real"]
                TR_real_EV_stra = cost_df_temp.loc[min_cost_idx_real]["ego_stra"]

                TR_index = cost_df_temp[
                    (cost_df_temp["ego_stra"] == EV_stra) & (cost_df_temp["TR_stra"] == TR_real_stra)
                    & (cost_df_temp["TF_stra"] == TF_stra)
                    ]["index"].values[0]
                traj_data_real = traj_data[TR_index]
                TR_esti_lambda = traj_data_opti.TR_cost_lambda
                TR_real_lambda = traj_data_real.TR_cost_lambda

            elif isinstance(TR, Game_A_Vehicle) and isinstance(TF, Game_H_Vehicle):
                # ego与TR的合作
                TR_strategy = TR.get_strategies(is_lc=False, single_stra=single_stra)
                ego_strategy = self.get_strategies(
                    is_lc=self.lc_direction != 0 or self.lane_changing,  # 意图或者已经换道
                    lc_after=self.lc_direction == 0  # 是否过线
                )

                cost_df, traj_data = self._cal_cost_df(
                    stra_product=itertools.product(ego_strategy, [None], TR_strategy, [None], [None]),
                    cal_TF_cost=False, cal_TR_cost=True,
                    TF=TF, TR=TR, PC=PC, CR=CR
                )

                cost_df["total_cost"] = cost_df["ego_cost"] + TR.game_co * cost_df["TR_cost_real"]  # 合作效用
                # cost_df_temp = cost_df[cost_df["total_cost"] != np.inf]
                # if len(cost_df_temp) == 0:
                #     continue
                cost_df_temp = cost_df
                min_cost_idx = cost_df_temp["total_cost"].idxmin()  # 找到最小的cost值
                # 更新TR的策略
                TR_real_stra = cost_df_temp.loc[min_cost_idx]["TR_stra"]
                EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                TR_real_cost = cost_df_temp.loc[min_cost_idx]["TR_cost_real"]
                EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
                EV_opti_series = cost_df_temp.loc[min_cost_idx]

                index = EV_opti_series["index"]
                traj_data_opti = traj_data[index]
            elif isinstance(TR, Game_A_Vehicle) and isinstance(TF, Game_A_Vehicle):
                # ego与TR的合作，ego与TF的合作
                TR_strategy = TR.get_strategies(is_lc=False, single_stra=single_stra)
                TF_strategy = TF.get_strategies(is_lc=False, single_stra=single_stra)
                ego_strategy = self.get_strategies(
                    is_lc=self.lc_direction != 0 or self.lane_changing,  # 意图或者已经换道
                    lc_after=self.lc_direction == 0  # 是否过线
                )

                cost_df, traj_data = self._cal_cost_df(
                    stra_product=itertools.product(ego_strategy, TF_strategy, TR_strategy, [None], [None]),
                    cal_TF_cost=True, cal_TR_cost=True,
                    TF=TF, TR=TR, PC=PC, CR=CR
                )

                cost_df["total_cost"] = (
                        cost_df["ego_cost"] + TR.game_co * cost_df["TR_cost_real"] + TF.game_co * cost_df[
                    "TF_cost_real"]
                )
                cost_df_temp = cost_df
                # cost_df_temp = cost_df[cost_df["total_cost"] != np.inf]
                # if len(cost_df_temp) == 0:
                #     continue
                min_cost_idx = cost_df_temp["total_cost"].idxmin()  # 找到最小的cost值

                TR_real_stra = cost_df_temp.loc[min_cost_idx]["TR_stra"]
                TF_stra = cost_df_temp.loc[min_cost_idx]["TF_stra"]
                EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                TR_real_cost = cost_df_temp.loc[min_cost_idx]["TR_cost_real"]
                TF_cost = cost_df_temp.loc[min_cost_idx]["TF_cost_real"]
                EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
                EV_opti_series = cost_df_temp.loc[min_cost_idx]

                index = EV_opti_series["index"]
                traj_data_opti: TrajData = traj_data[index]
            else:
                print(f"车辆类型错误：TR: {type(TR)}, TF: {type(TF)}")
                raise NotImplementedError("未知的车辆类型")

            if traj_data_opti.EV_traj is None:
                continue

            game_res_list.append(GameRes(
                self.lane.step_,
                cost_df, self, TF, TR, PC, CR,
                EV_stra=EV_stra, TF_stra=TF_stra, TR_stra=TR_real_stra, CR_stra=CR_stra, CP_stra=CP_stra,
                EV_cost=EV_cost, TF_cost=TF_cost, TR_cost=TR_real_cost, CR_cost=CR_cost, CP_cost=CP_cost,
                TR_real_EV_stra=TR_real_EV_stra,
                EV_opti_series=EV_opti_series, traj_data=traj_data_opti,
                EV_safe_cost=EV_opti_series.safe_cost,
                EV_com_cost=EV_opti_series.com_cost,
                EV_eff_cost=EV_opti_series.eff_cost,
                EV_route_cost=EV_opti_series.route_cost,
                TR_esti_lambda=TR_esti_lambda,
                TR_real_lambda=TR_real_lambda,
            ))

        return game_res_list

    def _cal_cost_df(
            self, stra_product, TF, TR, CR, PC,
            cal_TF_cost=False, cal_TR_cost=False, cal_CR_cost=False, cal_CP_cost=False
    ):
        stra_product = list(stra_product)

        cost_data = []
        traj_data = []
        # a0~a5, b0~b5, step, y1, T, is_lc, TF_stra, TR_stra, cost, TF_cost, TR_cost
        for i, [ego_stra, TF_stra, TR_stra, CR_stra, CP_stra] in enumerate(stra_product):
            result, traj = self.cal_cost_given_stra(
                ego_stra, TF_stra=TF_stra, TR_stra=TR_stra, CR_stra=CR_stra, CP_stra=CP_stra,
                TF=TF, TR=TR, PC=PC, CR=CR,
                cal_TF_cost=cal_TF_cost, cal_TR_cost=cal_TR_cost,
                cal_CR_cost=cal_CR_cost, cal_PC_cost=cal_CP_cost,
            )
            result.append(i)
            cost_data.append(result)
            traj_data.append(traj)

        # *lc_result[1:],
        # stra, TF_stra, TR_stra, CR_stra, CP_stra,
        # lc_result[0], TF_cost_hat, TR_cost_hat, CR_cost_hat, CP_cost_hat,
        # TF_real_cost, TR_real_cost, CR_real_cost, CP_real_cost
        cost_df = pd.DataFrame(
            cost_data,
            columns=[*[f"{i}{j}" for i in ["a", "b"] for j in range(6)], *["step", "y1"],
                     *["ego_stra", "TF_stra", "TR_stra", "CR_stra", "CP_stra",
                       'ego_cost', "TF_cost_hat", "TR_cost_hat", "CR_cost_hat", "CP_cost_hat",
                       'TF_cost_real', 'TR_cost_real', "CR_cost_real", "CP_cost_real",
                       "safe_cost", "com_cost", "eff_cost", "route_cost", "index"]],
        )
        return cost_df, traj_data

    def cal_cost_given_stra(
            self, stra, TF_stra=None, TR_stra=None, CR_stra=None, CP_stra=None,
            TF: Game_Vehicle = None, TR: Game_Vehicle = None, PC: Game_Vehicle = None,
            CR: Game_Vehicle = None,
            cal_TF_cost=False, cal_TR_cost=False, cal_PC_cost=False, cal_CR_cost=False,
    ):
        """根据策略计算效用值
        :return: a0~a5, b0~b5, step, y1, T, is_lc, TF_stra, TR_stra, TF_cost_hat, TR_cost_hat, cost,
         TF_cost_real, TR_cost_real,
        """
        if isinstance(stra, Iterable):
            T, is_lc = stra
        else:
            T, is_lc = stra, False
        # 换道效用
        lc_direction = self.lc_direction if is_lc else 0
        y1, y_limit_low, y_limit_up, y_middle = self.get_y_constraint(lc_direction)
        (lc_result, [TF_traj, TR_traj, PC_traj, CR_traj, TF_PC_traj, TR_PC_traj, PC_PC_traj, CR_PC_traj],
         [safe_cost, com_cost, eff_cost, route_cost]) = \
            self.opti_quintic_given_T(
                T, y1, y_limit_low, y_limit_up, lc_direction, y_middle,
                TF=TF, TR=TR, PC=PC, CR=CR,
                TF_stra=TF_stra, TR_stra=TR_stra, CP_stra=CP_stra, CR_stra=CR_stra,
                return_other_traj=True
            )

        x_opt = lc_result[1:13]
        times = np.arange(0, T + self.dt / 2, self.dt)
        if lc_result[0] != np.inf:
            EV_traj = np.vstack([np.array(get_xy_quintic(x_opt, t)) for t in times])
        else:
            EV_traj = None

        TF_rho_hat = TF.rho if isinstance(TF, Game_A_Vehicle) else self.get_rho_hat_s([TF])[0]
        TR_rho_hat = TR.rho if isinstance(TR, Game_A_Vehicle) else self.get_rho_hat_s([TR])[0]
        CP_rho_hat = PC.rho if isinstance(PC, Game_A_Vehicle) else self.get_rho_hat_s([PC])[0]
        CR_rho_hat = CR.rho if isinstance(CR, Game_A_Vehicle) else self.get_rho_hat_s([CR])[0]

        # TF
        TF_cost_hat, TF_real_cost, TF_cost_lambda = cal_other_cost(
            TF, cal_TF_cost, TF_rho_hat, TF_traj, [TF_PC_traj],
            v_length_s=[TR.length if TR is not None else self.length]
        )
        CP_cost_hat, CP_real_cost, CP_cost_lambda = cal_other_cost(
            PC, cal_PC_cost, CP_rho_hat, PC_traj,
            [PC_PC_traj],
            v_length_s=[PC.f.length if PC.f is not None else self.length]
        )
        if is_lc:
            # TR
            TR_cost_hat, TR_real_cost, TR_cost_lambda = cal_other_cost(
                TR, cal_TR_cost, TR_rho_hat, TR_traj,
                [EV_traj, TF_traj], v_length_s=[self.length, TF.length],
                route_cost=route_cost if EV_traj is not None else 0, print_cost=False
            )
            CR_cost_hat, CR_real_cost, CR_cost_lambda = cal_other_cost(
                CR, cal_CR_cost, CR_rho_hat, CR_traj,
                [EV_traj, PC_traj], v_length_s=[self.length, PC.length]
            )
        else:
            TR_cost_hat, TR_real_cost, TR_cost_lambda = cal_other_cost(
                TR, cal_TR_cost, TR_rho_hat, TR_traj,
                [TR_PC_traj], v_length_s=[TF.length],
                print_cost=False
            )
            CR_cost_hat, CR_real_cost, CR_cost_lambda = cal_other_cost(
                CR, cal_CR_cost, CR_rho_hat, CR_traj,
                [EV_traj], v_length_s=[self.length]
            )

        result = [
            *lc_result[1:],
            stra, TF_stra, TR_stra, CR_stra, CP_stra,
            lc_result[0], TF_cost_hat, TR_cost_hat, CR_cost_hat, CP_cost_hat,
            TF_real_cost, TR_real_cost, CR_real_cost, CP_real_cost,
            safe_cost, com_cost, eff_cost, route_cost,
        ]

        assert TR_cost_lambda is not None

        traj_data = TrajData(
            EV_traj, TF_traj, TR_traj, PC_traj, CR_traj,
            TF_PC_traj, TR_PC_traj, PC_PC_traj, CR_PC_traj,
            TF_cost_lambda, TR_cost_lambda, CP_cost_lambda, CR_cost_lambda,
        )
        return result, traj_data

    def opti_quintic_given_T(
            self, T, y1, y_limit_low, y_limit_up, lc_direction, y_middle=None,
            TF: Game_Vehicle = None, TR: Game_Vehicle = None, PC: Game_Vehicle = None, CR: Game_Vehicle = None,
            TF_stra=None, TR_stra=None, CP_stra=None, CR_stra=None,
            return_other_traj=False
    ):
        """[cost_value], x_opt, [T, step, y1]   [TF_traj, TR_traj, PC_traj]"""
        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x0
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # dx0
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ddx0
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # y0
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # dy0
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # ddy0
        ])

        constraints = []
        x = cvxpy.Variable(12 + 4)  # fxt，fyt的五次多项式系数 + 安全松弛变量

        A2 = np.array([
            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],  # y1
            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],  # dy1
            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]  # ddy1
        ])
        b2 = np.array([y1, 0, 0])

        constraints += [A @ x[: 12] == self.state]  # 初始状态约束
        constraints += [A2 @ x[6: 12] == b2]  # 终止状态约束

        # ddx1为0
        # A_ = np.array([
        #     [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3],  # ddx1
        # ])
        # b_ = np.array([0])
        # constraints += [A_ @ x[: 6] == b_]

        # 终止ax为0
        # A3 = np.array([
        #     [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3],  # ddx1
        # ])
        # constraints += [A3 @ x[: 6] == 0]

        time_steps = np.arange(0, T + self.dt / 2, self.dt)

        # 计算每个时间步的y约束
        A_ineq = np.array([
            [1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps
        ])
        b_ineq = np.array([y_limit_low] * len(A_ineq))
        b_ineq2 = np.array([y_limit_up] * len(A_ineq))
        constraints += [A_ineq @ x[6: 12] >= b_ineq]
        constraints += [A_ineq @ x[6: 12] <= b_ineq2]

        # 计算每个时间步的dx约束
        A_ineq3 = np.array([
            [0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps
        ])
        b_ineq3 = np.array([3] * len(A_ineq3))
        b_ineq4 = np.array([self.vel_desire + 5] * len(A_ineq3))
        constraints += [A_ineq3 @ x[: 6] >= b_ineq3]
        constraints += [A_ineq3 @ x[: 6] <= b_ineq4]

        # 计算每个时间步的ddx和ddy约束
        A_ineq5 = np.array([
            [0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3] for t in time_steps
        ])
        b_ineq5 = np.array([self.acc_max] * len(A_ineq5))
        b_ineq6 = np.array([- self.dec_max] * len(A_ineq5))
        constraints += [A_ineq5 @ x[6: 12] <= b_ineq5]
        constraints += [A_ineq5 @ x[6: 12] >= b_ineq6]
        constraints += [A_ineq5 @ x[: 6] <= b_ineq5]
        constraints += [A_ineq5 @ x[: 6] >= b_ineq6]

        # 计算换道关键时间点的安全约束
        # 预测换道结束时目标车道前后车辆位置
        vehicles = [TF, TR, PC, CR]
        ((TF_traj, l_TF, TF_PC_traj), (TR_traj, l_TR, TR_PC_traj),
         (PC_traj, l_PC, PC_PC_traj), (CR_traj, l_CR, CR_PC_traj)) = \
            self.pred_traj_s(
                T,
                vehicles,
                stra_s=[TF_stra, TR_stra, CP_stra, CR_stra],
            )

        xt_TF, dxt_TF = TF_traj[:, 0], TF_traj[:, 1]
        xt_TR, dxt_TR = TR_traj[:, 0], TR_traj[:, 1]
        xt_PC, dxt_PC = PC_traj[:, 0], PC_traj[:, 1]
        xt_CR, dxt_CR = CR_traj[:, 0], CR_traj[:, 1]

        if lc_direction == 0:
            # PC
            dxt = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps])
            A_ineq_7 = np.array([
                [1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])  # 纵向位置
            x_pos = A_ineq_7 @ x[: 6]  # 纵向位置
            v = dxt @ x[:6]
            PC_rear_x = xt_PC - l_PC
            b_ineq_7_part1 = PC_rear_x - (v * self.time_safe + self.safe_s0) * self.SCALE
            constraints += [x_pos <= b_ineq_7_part1]
            b_ineq_7_part2 = PC_rear_x - v * self.time_wanted * self.SCALE
            constraints += [x[12] >= (x_pos - b_ineq_7_part2) / (self.state[1] * self.time_wanted * self.SCALE)]

            # CR
            CR_rear_x = xt_CR + l_CR
            b_ineq_8_part1 = CR_rear_x + (CR.time_safe * dxt_CR + self.safe_s0) * self.SCALE
            constraints += [x_pos >= b_ineq_8_part1]
            b_ineq_8_part2 = CR_rear_x + CR.time_wanted * dxt_CR * self.SCALE
            constraints += [x[13] >= (b_ineq_8_part2 - x_pos) / (dxt_CR[0] * CR.time_wanted * self.SCALE)]

            # 安全性
            constraints += [x[14] == 0]
            constraints += [x[15] == 0]
            # safe_cost = x[12] + cvxpy.abs(v[-1] - dxt_PC[-1]) + cvxpy.abs(v[-1] - dxt_CR[-1])
            safe_cost = cvxpy.max(cvxpy.hstack([[-1], x[12: 14]]))

            # 终端平顺性
            # ddx1为0
            A_ = np.array([
                0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3  # ddx1
            ])
            end_cost = (A_ @ x[: 6] - dxt_PC[-1]) / 10

            step = np.nan
        else:
            # 提取换道至目标车道初始时刻
            y, yf, dyf, ddyf = get_y_guess(T, self.state, y1)
            A_y = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])
            Y = A_y @ y
            # 进入目标车道的初始时刻 # ATTENTION：由于为车头的中点的y坐标，因此初始时刻为近似
            if Y[0] < Y[-1]:
                step = (Y[Y <= y_middle]).shape[0]  # 进入目标车道的初始时刻
            else:
                step = (Y[Y >= y_middle]).shape[0]  # 进入目标车道的初始时刻
            lc_before_times = time_steps[:step]
            lc_after_times = time_steps[step:]

            A_ineq_7 = np.array([
                [1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in lc_before_times])  # 过线点前位置
            x_pos_before = A_ineq_7 @ x[: 6]  # 过线点前位置
            A_ineq_8 = np.array([
                [1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in lc_after_times])  # 过线点后位置
            x_pos_after = A_ineq_8 @ x[: 6]  # 过线点后位置

            # PC
            dxt_before = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in lc_before_times])
            v_before = dxt_before @ x[:6]
            PC_rear_x = xt_PC[:step] - l_PC
            b_ineq_7_part1 = PC_rear_x - (v_before * self.time_safe + self.safe_s0) * self.SCALE
            constraints += [x_pos_before <= b_ineq_7_part1]
            b_ineq_7_part2 = PC_rear_x - v_before * self.time_wanted * self.SCALE
            constraints += [x[12] >= (x_pos_before - b_ineq_7_part2) / (self.state[1] * self.time_wanted * self.SCALE)]

            # CR
            CR_head_x = xt_CR[:step] + l_CR
            CR_v = dxt_CR[:step]
            b_ineq_10_part1 = CR_head_x + (CR.time_safe * CR_v + self.safe_s0) * self.SCALE
            constraints += [x_pos_before >= b_ineq_10_part1]
            b_ineq_10_part2 = CR_head_x + CR.time_wanted * CR_v * self.SCALE
            constraints += [x[13] >= (b_ineq_10_part2 - x_pos_before) / (CR_v[0] * CR.time_wanted * self.SCALE)]

            # TF
            dxt_after = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in lc_after_times])
            v_after = dxt_after @ x[:6]
            TF_rear_x = xt_TF[step:] - l_TF
            b_ineq_8_part1 = TF_rear_x - (v_after * self.time_safe + self.safe_s0) * self.SCALE
            constraints += [x_pos_after <= b_ineq_8_part1]
            b_ineq_8_part2 = TF_rear_x - v_after * self.time_wanted * self.SCALE
            constraints += [x[14] >= (x_pos_after - b_ineq_8_part2) / (self.state[1] * self.time_wanted * self.SCALE)]

            # TR
            TR_head_x = xt_TR[step:] + l_TR
            TR_v = dxt_TR[step:]
            b_ineq_9_part1 = TR_head_x + (TR.time_safe * TR_v + self.safe_s0) * self.SCALE
            constraints += [x_pos_after >= b_ineq_9_part1]
            b_ineq_9_part2 = TR_head_x + TR.time_wanted * TR_v * self.SCALE
            constraints += [x[15] >= (b_ineq_9_part2 - x_pos_after) / (TR_v[0] * TR.time_wanted * self.SCALE)]

            # 安全性
            safe_cost = cvxpy.max(cvxpy.hstack([[-1], x[12: 16]]))

            # 终端平顺性
            # ddx1为0
            A_ = np.array([
                0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3  # ddx1
            ])
            end_cost = (A_ @ x[: 6] - dxt_TF[-1]) / 10

        # 终止ax
        A3 = np.array([
            0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3,  # ddx1
        ])
        end_cost += (A3 @ x[: 6]) / self.acc_max

        # 舒适性：横纵向jerk目标
        A_xy_jerk = np.array([
            [0, 0, 0, 6, 24 * t, 60 * t ** 2] for t in time_steps
        ])
        # com_cost = 0
        com_cost = 0.5 * cvxpy.max(cvxpy.abs(A_xy_jerk @ x[:6]))
        com_cost += 0.5 * cvxpy.max(cvxpy.abs(A_xy_jerk @ x[6:12]))
        com_cost = com_cost / self.JERK_MAX + end_cost

        # 效率
        A_dxy = np.array([
            [0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps
        ])
        vx = A_dxy @ x[:6]
        eff_cost = (vx[0] - cvxpy.mean(vx)) / 10

        _, route_cost = self.cal_route_cost(self.lane.index + lc_direction, self.x)
        _, route_ori = self.cal_route_cost(self.lane.index, self.x)
        route_cost = route_cost - route_ori
        # A_ineq_7 = np.array([
        #     [1, T, T ** 2, T ** 3, T ** 4, T ** 5]])  # 纵向位置
        # x_pos = A_ineq_7 @ x[: 6]  # 纵向位置
        # route_cost += (x_pos[-1] - x_pos[0]) / (10 * self.state[1])
        # route_cost = route_cost * 4

        safe_cost = self.k_s * safe_cost
        com_cost = self.k_c * com_cost
        eff_cost = self.k_e * eff_cost
        route_cost = self.k_r * route_cost

        cost = ((1 - self.rho) * (
                safe_cost + com_cost
        ) + self.rho * eff_cost + route_cost)

        result = np.hstack([[np.inf], np.tile(np.nan, 12), [np.nan, y1]])
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        try:
            cost_value = prob.solve(verbose=False, solver=cvxpy.GUROBI, reoptimize=True)
            if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
                x_opt = x.value
                result = np.hstack([[cost_value], x_opt[:12], [step, y1]])
                # if target_direction != 0:
                #     print(f"换道方向：{target_direction}, 换道时长：{T}, 目标横向位置：{y1}")
        except cvxpy.error.SolverError:
            pass

        safe_cost_value = safe_cost.value if safe_cost is not None else np.nan
        com_cost_value = com_cost.value if com_cost is not None else np.nan
        eff_cost_value = eff_cost.value if eff_cost is not None else np.nan
        if hasattr(route_cost, "value"):
            route_cost_value = route_cost.value if route_cost is not None else np.nan
        else:
            route_cost_value = route_cost

        if return_other_traj:
            other_traj = [TF_traj, TR_traj, PC_traj, CR_traj, TF_PC_traj, TR_PC_traj, PC_PC_traj, CR_PC_traj]
            return result, other_traj, [safe_cost_value, com_cost_value, eff_cost_value, route_cost_value]
        return result

    @staticmethod
    def pred_traj_s(T, vehicles: list[Game_Vehicle], stra_s) -> list[tuple[np.ndarray, float, np.ndarray]]:
        pred_traj_s = []
        for v, stra in zip(vehicles, stra_s):
            # v = v if v is not None else self.f if self.lc_direction == 1 else self.r
            if v is None:
                pred_traj_s.append((None, None, None))
                continue
            l = v.length
            traj, PC_traj = v.pred_self_traj(T, stra, to_ndarray=True, ache=True)  # ATTENTION
            traj: np.ndarray = traj
            PC_traj: np.ndarray = PC_traj
            pred_traj_s.append((traj, l, PC_traj))
        return pred_traj_s

    def cal_ref_path(self, x_opt):
        times = np.arange(0, (self.opti_game_res.EV_lc_step + 0.5) * self.dt, self.dt)
        # x, vx, ax, y, vy, ay
        ref_path_ori = np.vstack([np.array(get_xy_quintic(x_opt, t)) for t in times])
        lc_end_state = ref_path_ori[-1, :]
        x, vx, ax, y, vy, ay = list(lc_end_state)
        times_after = np.arange(self.dt, (self.N_MPC + 0.5) * self.dt, self.dt).reshape(-1, 1)
        ref_path = np.vstack(
            [
                ref_path_ori,
                np.hstack([
                    x + vx * times_after,
                    np.tile(vx, (self.N_MPC, 1)),
                    np.tile(ax, (self.N_MPC, 1)),
                    np.tile(y, (self.N_MPC, 1)),
                    np.tile(vy, (self.N_MPC, 1)),
                    np.tile(ay, (self.N_MPC, 1))
                ])
            ]
        )  # 填补参考轨迹
        self.opti_game_res.EV_opti_traj = ref_path
        ref_path = ReferencePath(ref_path, self.dt)
        self.mpc_solver = MPC_Solver(self.N_MPC, ref_path, self)
        self.mpc_solver.init_mpc()
        return ref_path_ori

    def pred_self_traj(self, time_len, stra=None,
                       target_lane: "LaneAbstract" = None,
                       PC_traj=None, to_ndarray=True, ache=False, PC=None):
        """在策略下预测自车轨迹（包含初始状态）
        :param time_len: 预测时间长度
        :param stra: 期望时距
        :param target_lane: 目标车道
        :param PC_traj: 前车轨迹
        :param to_ndarray: 是否转为ndarray
        :param ache: 是否缓存
        """
        if self.is_game_leader and self.mpc_solver is not None:
            traj_points: list[TrajPoint] = self.mpc_solver.ref_path.get_ref_pos(self.mpc_solver.step, time_len)
            for point in traj_points:
                point.length = self.length
                point.width = self.width
            return traj_points, None

        traj, PC_traj = super().pred_self_traj(
            time_len, stra, target_lane, PC_traj, to_ndarray=to_ndarray, ache=ache, PC=PC
        )
        return traj, PC_traj


class Game_H_Vehicle(Game_Vehicle):
    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)

    def cal_vehicle_control(self):
        if self.opti_gap is not None:
            if self.opti_gap.adapt_end_time > self.lane.time_:
                self.next_acc = self.opti_gap.target_acc
            else:
                self.target_lane = self.opti_gap.target_lane if self.opti_gap is not None else self.lane
                veh_surr = VehSurr(ev=self, cp=self.opti_gap.TF)
                self.next_acc = self.cf_model.step(veh_surr)
        else:
            next_acc_block = np.inf
            if self.lane.index not in self.destination_lane_indexes:
                next_acc_block = self.cf_model.step(VehSurr(ev=self, cp=self.lane.road.end_weaving_block_veh))
            next_acc = self.cf_model.step(self.pack_veh_surr())
            self.next_acc = min(next_acc, next_acc_block)

        self.next_delta = self.cf_lateral_control()
        # print("ID:", self.ID, "lane_changing", self.lane_changing,
        #       "lc_end_step", self.lc_end_step, "current_step", self.lane.step_,
        #       'acc', self.next_acc, 'delta', self.next_delta, "opti_gap", self.opti_gap)
        return self.next_acc, self.next_delta


class Game_O_Vehicle(Game_H_Vehicle):
    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)

    def get_strategies(self, is_lc: bool = False, single_stra=False, lc_after=False):
        return [1]

    def pred_self_traj(self, time_len, stra=None,
                       target_lane: "LaneAbstract" = None,
                       PC_traj=None, to_ndarray=True, ache=False, PC: Vehicle = None):
        step_num = round(time_len / self.dt) + 1
        if to_ndarray:
            traj_list = [self.get_traj_point().to_ndarray()] * step_num
        else:
            traj_list = [self.get_traj_point()] * step_num
        return np.array(traj_list), None
