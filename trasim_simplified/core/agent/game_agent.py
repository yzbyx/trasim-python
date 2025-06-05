# -*- coding: utf-8 -*-
# @time : 2025/3/22 22:13
# @Author : yzbyx
# @File : game_agent.py
# Software: PyCharm
import abc
import itertools
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import sympy as sp
from sympy import Interval

from trasim_simplified.core.agent.base_agent import Base_Agent
from trasim_simplified.core.agent.game_help_func import cal_other_cost, get_TR_real_stra, get_CR_real_stra
from trasim_simplified.core.agent.mpc_solver import MPC_Solver
from trasim_simplified.core.agent.opti_quintic import opti_quintic_given_T_single, opti_quintic_given_T_weaving, \
    opti_quintic_given_T_platoon
from trasim_simplified.core.agent.ref_path import ReferencePath
from trasim_simplified.core.agent.utils import interval_intersection
from trasim_simplified.core.constant import VehSurr, GameRes, TrajData, V_TYPE, GameVehSurr, StraInfo, SolveRes

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class Game_Vehicle(Base_Agent, abc.ABC):
    NAME = "Game-V"

    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        # self.cf_stra_s = [0.6, 0.8, 1, 1.2, 1.4]
        self.cf_stra_s = [0.6, 1, 1.4]
        self.stra_times = np.arange(1, 10.1, 3)
        self.DELTA_V = 10
        self.LC_TIME_BASE = 10  # 换道时间基准
        self.LC_VEL_INCREASE_TOL = 5  # 换道速度增加容忍度

    def get_rho_hat(self, vehicle):
        if vehicle.ID not in self.rho_hat_s:
            self.rho_hat_s[vehicle.ID] = [0, 1]
        rho_hat = np.mean(self.rho_hat_s[vehicle.ID])
        return rho_hat

    def set_stra(self, stra_info: StraInfo):
        """设置策略"""
        if stra_info is None:
            return
        self.is_gaming = True
        self.is_game_leader = isinstance(self, Game_A_Vehicle)
        self.cf_factor = stra_info.cf_stra
        self.target_lane = stra_info.target_lane
        self.lc_direction = stra_info.lc_direction
        if self.lc_direction != 0:
            self.lane_changing = True

    def clear_lc_state(self):
        self.target_lane = self.lane
        self.lc_direction = 0
        self.lane_changing = False

    def clear_stra(self, clear_lc_state=True, reason=""):
        """清除策略"""
        self.is_gaming = False
        self.is_game_leader = False
        self.is_game_initiator = False
        self.cf_factor = 1
        self.clear_lc_state() if clear_lc_state else None

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
                         route_cost=0, return_sub_cost=False, print_cost=False,
                         stra_info: StraInfo = None):
        """
        :param stra_info:
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
        # 舒适性计算
        jerk_x_max = max(abs(np.diff(traj[:, 2])))
        jerk_y_max = max(abs(np.diff(traj[:, 5])))

        com_cost = self.k_c * (0.5 * jerk_x_max + 0.5 * jerk_y_max) / self.JERK_MAX  # 舒适性计算
        # 效率计算 单位时间内的平均速度-初始速度
        vx = traj[:, 1]
        eff_cost = self.k_e * (vx[0] - np.mean(vx)) / self.DELTA_V  # 效率计算

        route_cost = route_cost  # 路径成本

        def cost(rho_, is_print=False):
            total_cost = (
                    (1 - rho_) * (safe_cost + com_cost + route_cost) +
                    rho_ * eff_cost
            )
            if is_print:
                print("call lambda", rho_, safe_cost, com_cost, eff_cost, route_cost)
            return total_cost

        if stra_info.solve_res is None:
            T = (traj.shape[0] - 1) * self.lane.dt
            times = np.arange(0, T + self.dt / 2, self.dt)
            stra_info.solve_res = SolveRes(
                None, times, safe_cost, com_cost, eff_cost, route_cost, cost(self.rho), cost_lambda=cost
            )
            stra_info.solve_res.set_traj(traj)

        if print_cost:
            print("traj_cost:", safe_cost, com_cost, eff_cost, route_cost)

        if return_lambda:
            return cost

        if return_sub_cost:
            return safe_cost, com_cost, eff_cost, route_cost

        return cost(rho)

    @abc.abstractmethod
    def get_strategies(self, game_surr: GameVehSurr, single_stra=False) -> list[StraInfo]:
        """
        返回可能的策略（持续时间, 换道方向）
        :param game_surr:
        :param single_stra: 是否单策略
        :return: （持续时间, 策略，是否换道）
        """
        pass

    def _make_dummy_agent(self, lane, type_, id_, length, x, y):
        dummy_agent = Game_O_Vehicle(lane, type_, id_, length)
        dummy_agent.x = x
        dummy_agent.y = y
        dummy_agent.speed = self.speed
        dummy_agent.cf_model = self.cf_model
        dummy_agent.lc_model = self.lc_model
        return dummy_agent


class Game_A_Vehicle(Game_Vehicle):
    """自动驾驶车辆"""
    rho_hat_s = {}  # 储存车辆id_对应的rho_hat范围
    NAME = "Game-AV"

    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.N_MPC = 5
        self.mpc_solver: Optional[MPC_Solver] = None
        self.last_cal_step = -1
        self.re_cal_step = 10
        self.decision_frequency = 10

    def get_strategies(self, game_surr: GameVehSurr, single_stra=False) -> list[StraInfo]:
        if single_stra:
            return [StraInfo(self, None, 1, 0)]
        if game_surr.EV.ID == self.ID:
            cf_stra = list(itertools.product(self.stra_times, [1], [0]))
            lc_stra = [(stra_time, 1, game_surr.lc_direction) for stra_time in self.stra_times]
            if game_surr.lc_direction == 0:  # 保持车道
                strategy = cf_stra
            else:
                strategy = lc_stra + cf_stra
        elif (self.route_lc_direction != 0 and game_surr.TR.ID == self.ID and
              game_surr.lc_direction == - self.route_lc_direction):  # 被动lc
            # 交织换道
            print(f"{self.name} get_strategies, weaving")
            lc_stra = [(None, 1, - game_surr.lc_direction)]
            cf_stra = list(itertools.product([None], self.cf_stra_s, [0]))
            strategy = lc_stra + cf_stra
            game_surr.TR.no_lc = False  # 被动换道的情况下，TR允许换道
        elif (self.route_lc_direction != 0 and game_surr.CR.ID == self.ID and
              game_surr.lc_direction == self.route_lc_direction):  # 被动lc
            # 队列换道
            print(f"{self.name} get_strategies, platoon")
            lc_stra = [(None, 1, game_surr.lc_direction)]
            cf_stra = list(itertools.product([None], self.cf_stra_s, [0]))
            strategy = lc_stra + cf_stra
            game_surr.CR.no_lc = False  # 被动换道的情况下，CR允许换道
        else:
            strategy = list(itertools.product([None], self.cf_stra_s, [0]))

        strategy = [StraInfo(self, stra_time, stra, lc_direction) for stra_time, stra, lc_direction in strategy]
        return strategy

    def cal_vehicle_control(self):
        """横向控制"""
        # if self.lane_changing is False or self.lane == self.target_lane:
        if self.mpc_solver is None or self.mpc_solver.is_end:
            delta = self.cf_lateral_control()
            acc = self.cf_model.step(self.pack_veh_surr())
            next_acc_block = np.inf
            if self.lane.index not in self.destination_lane_indexes:
                next_acc_block = self.cf_model.step(VehSurr(ev=self, cp=self.lane.road.end_weaving_block_veh))
            acc = min(acc, next_acc_block)
        else:
            # 横向控制
            acc, delta, is_end = self.mpc_solver.step_mpc()  # 默认计算出的符合约束的加速度和转向角
            if is_end:
                self.lane_changing = False
                self.lc_conti_time = 0
                print(f"{self.name} MPC end")

        self.next_acc = acc
        self.next_delta = delta

    def update_rho_hat(self):
        """求解rho的范围，入股TR的rho_hat不在范围内，更新TR的rho_hat"""
        TR = self.opti_game_res.game_surr.TR
        TR_est_lambda = self.opti_game_res.TR_esti_lambda
        TR_real_lambda = self.opti_game_res.TR_real_lambda

        CR = self.opti_game_res.game_surr.CR
        CR_est_lambda = self.opti_game_res.CR_esti_lambda
        CR_real_lambda = self.opti_game_res.CR_real_lambda

        for veh, esti_lambda, real_lambda in [(TR, TR_est_lambda, TR_real_lambda), (CR, CR_est_lambda, CR_real_lambda)]:
            if isinstance(veh, Game_H_Vehicle):
                # 构建不等式求解问题
                rho_real = sp.symbols("rho_real")
                # 定义不等式
                inequality = sp.Le(real_lambda(rho_real), esti_lambda(rho_real))  # f(x) <= g(x)
                # 求解不等式
                solution = sp.solve_univariate_inequality(
                    inequality, rho_real, relational=False, domain=Interval(0, 1)
                )

                # 获取区间的上下限
                try:
                    lower_bound = float(solution.start)
                    upper_bound = float(solution.end)
                except:
                    lower_bound = 0
                    upper_bound = 1
                print("cal rho_hat", lower_bound, upper_bound)
                # 更新TR的rho_hat
                rho_hat_range_before = self.rho_hat_s[veh.ID]
                # 取交集
                rho_hat_range = interval_intersection(
                    rho_hat_range_before,
                    (lower_bound, upper_bound), print_flag=True
                )
                if rho_hat_range is not None:
                    self.rho_hat_s[veh.ID] = rho_hat_range

    def lc_intention_judge(self):
        if self.lane == self.target_lane and self.is_keep_lane_center():
            self.clear_stra(reason="keep lane center")

        if self.lane_changing and self.lane == self.target_lane:
            self.lc_direction = 0

        if not self.lane_changing:
            super().lc_intention_judge()

    def lc_decision_making(self, **kwargs):
        """判断是否换道"""
        super().lc_decision_making(set_lane_changing=False)
        if self.no_lc or not self.can_raise_game:
            return
        if self.is_game_initiator and self.last_cal_step == 1 and self.lane is not None:
            self.update_rho_hat()
        # LC轨迹优化，如果博弈选择不换道，需要重新更新self.lane_changing，is_gaming以及target_lane
        if (
                (self.last_cal_step >= self.re_cal_step)  # 换道过程重规划
                or (not self.lane_changing and self.lc_direction != 0)  # 未换道但有换道意图
                or (not self.is_keep_lane_center() and not self.lane_changing)
        ):
            if self.is_game_initiator or (not self.is_gaming):  # 必须是发起者，或者并未处于博弈（主要考虑换道过程重规划）
                print(f"{self.name} re_cal stackel_berg")
                self.stackel_berg()

        if self.lane_changing:
            self.lc_conti_time += self.lane.dt
            self.last_cal_step += 1

    def stackel_berg(self):
        """基于stackelberg主从博弈理论的换道策略，计算得到参考轨迹（无论是否换道）"""
        # 计算不同策略（换道时间）的最优轨迹和对应效用值
        # cal_game_matrix需要更新周边车辆的策略，stackel_berg函数只更新自身的策略
        self.game_res_list = self.cal_game_matrix()
        game_cost_list = [res.total_cost for res in self.game_res_list]
        if len(game_cost_list) == 0:
            self.clear_stra(reason="no feasible game result")
            return
        # 获取最优策略
        min_cost_idx = np.argmin(game_cost_list)
        opti_game_res = self.game_res_list[min_cost_idx]

        if opti_game_res.EV_stra.lc_direction == 0 and not self.lane_changing:
            return

        self.opti_game_res: GameRes = opti_game_res
        print(f"{self.name} set opti_game_res")
        self.set_stra(self.opti_game_res.EV_stra)

    def set_stra(self, stra_info: StraInfo):
        super().set_stra(stra_info)
        if self.opti_game_res is not None:
            # print(f"have opti_game_res {self.ID}")
            TR = self.opti_game_res.game_surr.TR
            TR.set_stra(self.opti_game_res.TR_stra)
            TP = self.opti_game_res.game_surr.TP
            TP.set_stra(self.opti_game_res.TF_stra)
            CR = self.opti_game_res.game_surr.CR
            CR.set_stra(self.opti_game_res.CR_stra)
            self.last_cal_step = 0
            self.is_game_initiator = True
        # self.set_path_mpc(stra_info.solve_res) if stra_info.lc_direction != 0 else None
        self.set_path_mpc(stra_info.solve_res) if not self.is_keep_lane_center() else None

    def clear_stra(self, clear_lc_state=True, reason=""):
        super().clear_stra(clear_lc_state=clear_lc_state)
        if self.opti_game_res is not None:
            TR = self.opti_game_res.game_surr.TR
            TR.clear_stra(reason="ev game end") if not TR.lane_changing else None
            CR = self.opti_game_res.game_surr.CR
            CR.clear_stra(reason="ev game end") if not CR.lane_changing else None
            TP = self.opti_game_res.game_surr.TP
            TP.clear_stra(reason="ev game end")
            if isinstance(TR, Game_A_Vehicle) and TR.lane_changing:
                TR.is_game_initiator = True
            elif isinstance(CR, Game_A_Vehicle) and CR.lane_changing:
                CR.is_game_initiator = True
        if clear_lc_state:
            self.mpc_solver = None
            self.lc_conti_time = 0
            self.last_cal_step = -1
            self.opti_game_res = None
        print(f"{self.name} clear strategy {reason=}")

    def cal_game_matrix(self):
        """backward induction method
        :return: opti_df (a0~a5, b0~b5, ego_stra, TF_stra, TR_stra, PC_stra,
         ego_cost, TP_cost, TR_cost, PC_cost)
        """
        game_res_list = []
        for gap in [-1, 0, 1]:
        # for gap in [0]:
            TR, TP, CP, CR, TRR, CRR, TPP, CPP = self._no_car_correction(gap, self.lc_direction, return_RR=True)
            game_surr = GameVehSurr(
                EV=self, TR=TR, TP=TP, CP=CP, CR=CR, TRR=TRR, CRR=CRR, TPP=TPP, CPP=CPP,
                lc_direction=self.lc_direction
            )

            ego_strategy = self.get_strategies(game_surr)
            # TR_strategy = TR.get_strategies(game_surr, single_stra=self.lc_direction == 0)
            # CR_strategy = CR.get_strategies(game_surr, single_stra=self.lc_direction == 0)
            TP_strategy = TP.get_strategies(game_surr, single_stra=True if isinstance(TP, Game_H_Vehicle) else False)
            CP_strategy = CP.get_strategies(game_surr, single_stra=True if isinstance(CP, Game_H_Vehicle) else False)
            TR_strategy = TR.get_strategies(game_surr)
            CR_strategy = CR.get_strategies(game_surr)

            cost_df, traj_data = self.cal_cost_df(
                lc_surr=game_surr,
                stra_product=itertools.product(ego_strategy, TP_strategy, TR_strategy, CR_strategy, CP_strategy)
            )

            # 根据估计的TR成本函数计算最小的当前车辆cost
            cost_df_temp = cost_df
            if len(cost_df_temp) == 0:
                continue
            # 找到最小的TR_cost值
            group_name = ["ego_stra", "TP_stra", "CP_stra"]
            if isinstance(TR, Game_A_Vehicle):
                group_name.append("TR_stra")
            if isinstance(CR, Game_A_Vehicle):
                group_name.append("CR_stra")

            if isinstance(TR, Game_A_Vehicle) and isinstance(CR, Game_A_Vehicle):
                min_cost_idx = cost_df_temp["total_cost"].idxmin()
            else:
                if isinstance(TR, Game_A_Vehicle) and not isinstance(CR, Game_A_Vehicle):
                    min_other_cost_idx = cost_df_temp.groupby(group_name)["CR_cost_hat"].idxmin()
                else:
                    min_other_cost_idx = cost_df_temp.groupby(group_name)["TR_cost_hat"].idxmin()
                min_cost_idx = cost_df_temp.loc[min_other_cost_idx].groupby(group_name)["total_cost"].idxmax()
                min_cost_idx = cost_df_temp.loc[min_cost_idx]["total_cost"].idxmin()  # 找到最小的cost值

            if isinstance(min_cost_idx, pd.Series):
                min_cost_idx = cost_df_temp.loc[min_cost_idx]["stra_time"].idxmin()

            EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
            EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
            CR_stra_hat = cost_df_temp.loc[min_cost_idx]["CR_stra"]
            TR_stra_hat = cost_df_temp.loc[min_cost_idx]["TR_stra"]
            TP_stra = cost_df_temp.loc[min_cost_idx]["TP_stra"]
            TP_cost = cost_df_temp.loc[min_cost_idx]["TP_cost_hat"]
            CP_stra = cost_df_temp.loc[min_cost_idx]["CP_stra"]
            CP_cost = cost_df_temp.loc[min_cost_idx]["CP_cost_hat"]

            stra_time = cost_df_temp.loc[min_cost_idx]["stra_time"]

            EV_opti_series = cost_df_temp.loc[min_cost_idx]
            index = EV_opti_series["index"]
            try:
                traj_data_opti = traj_data[index]
            except IndexError as e:
                print(index)
                raise e
            total_cost = cost_df_temp.loc[min_cost_idx]["total_cost"]
            cost_df_temp = cost_df_temp[cost_df_temp["stra_time"] == stra_time]

            TR_stra, TR_cost, TR_esti_lambda, TR_real_lambda = \
                get_TR_real_stra(cost_df_temp, EV_stra, CR_stra_hat, TP_stra, CP_stra, traj_data_opti, traj_data)
            CR_stra, CR_cost, CR_esti_lambda, CR_real_lambda = \
                get_CR_real_stra(cost_df_temp, EV_stra, TR_stra_hat, TP_stra, CP_stra, traj_data_opti, traj_data)

            if traj_data_opti.EV_traj is None:
                continue

            game_res_list.append(GameRes(
                self.lane.step_, game_surr, total_cost=total_cost,
                EV_stra=EV_stra, TF_stra=TP_stra, TR_stra=TR_stra, CR_stra=CR_stra, CP_stra=CP_stra,
                EV_cost=EV_cost, TP_cost=TP_cost, TR_cost=TR_cost, CR_cost=CR_cost, CP_cost=CP_cost,
                traj_data=traj_data_opti,
                TR_esti_lambda=TR_esti_lambda,
                TR_real_lambda=TR_real_lambda,
                CR_esti_lambda=CR_esti_lambda,
                CR_real_lambda=CR_real_lambda,
            ))

        return game_res_list

    def cal_cost_df(self, lc_surr: GameVehSurr, stra_product):
        stra_product = list(stra_product)

        cost_data = []
        traj_data = []
        count = 0
        for [ego_stra, TP_stra, TR_stra, CR_stra, CP_stra] in stra_product:
            result, traj = self.cal_cost_given_stra(
                ego_stra, TP_stra=TP_stra, TR_stra=TR_stra, CR_stra=CR_stra, CP_stra=CP_stra,
                lc_surr=lc_surr
            )
            if result is None:
                continue
            result.append(count)
            cost_data.append(result)
            traj_data.append(traj)
            count += 1

        cost_df = pd.DataFrame(
            cost_data,
            columns=["stra_time", "ego_stra", "TP_stra", "TR_stra", "CR_stra", "CP_stra", "total_cost",
                     'ego_cost', "TP_cost_hat", "TR_cost_hat", "CR_cost_hat", "CP_cost_hat",
                     'TP_cost_real', 'TR_cost_real', "CR_cost_real", "CP_cost_real", "index"],
        )
        return cost_df, traj_data

    def cal_cost_given_stra(
            self, EV_stra: StraInfo, TP_stra: StraInfo = None, TR_stra: StraInfo = None,
            CR_stra: StraInfo = None, CP_stra: StraInfo = None,
            lc_surr: Optional[GameVehSurr] = None
    ) -> tuple[Optional[list], Optional[TrajData]]:
        """根据策略计算效用值
        :return: a0~a5, b0~b5, step, y1, T, is_lc, TF_stra, TR_stra, TF_cost_hat, TR_cost_hat, cost,
         TF_cost_real, TR_cost_real,
        """
        T, lc_direction = EV_stra.stra_time, EV_stra.lc_direction

        TP, TR, CP, CR, CRR, TRR, CPP, TPP = \
            lc_surr.TP, lc_surr.TR, lc_surr.CP, lc_surr.CR, lc_surr.CRR, lc_surr.TRR, lc_surr.CPP, lc_surr.TPP

        # 若换道方向为0，则其他车辆的cf策略必须为1
        if EV_stra.lc_direction == 0:
            if (
                TP_stra.lc_direction != 0
                or TR_stra.lc_direction != 0
                or CR_stra.lc_direction != 0
                or CP_stra.lc_direction != 0
            ):
                return None, None
            if (
                EV_stra.cf_stra != 1
                or TP_stra.cf_stra != 1
                or TR_stra.cf_stra != 1
                or CR_stra.cf_stra != 1
                or CP_stra.cf_stra != 1
            ):
                return None, None

        EV_stra = EV_stra.copy()
        TP_stra = TP_stra.copy() if TP_stra is not None else None
        TR_stra = TR_stra.copy() if TR_stra is not None else None
        CR_stra = CR_stra.copy() if CR_stra is not None else None
        CP_stra = CP_stra.copy() if CP_stra is not None else None

        # 预测换道结束时目标车道前后车辆位置
        for stra_inf in [TP_stra, TR_stra, CR_stra, CP_stra]:
            stra_inf.stra_time = T
        if TR_stra.lc_direction != 0:
            TR_stra.veh_cp = CP
        if CR_stra.lc_direction != 0:
            CR_stra.veh_cp = TP
        if EV_stra.lc_direction != 0:
            CR_stra.veh_cp = CP
        TP_traj, _ = TP.pred_self_traj(TP_stra, to_ndarray=True, ache=True)
        TR_traj, _ = TR.pred_self_traj(TR_stra, to_ndarray=True, ache=True)
        CP_traj, _ = CP.pred_self_traj(CP_stra, to_ndarray=True, ache=True)
        CR_traj, _ = CR.pred_self_traj(CR_stra, to_ndarray=True, ache=True)
        CRR_traj = CRR.pred_net.pred_traj(CRR.pack_veh_surr(), time_len=T, to_ndarray=True)
        TRR_traj = TRR.pred_net.pred_traj(TRR.pack_veh_surr(), time_len=T, to_ndarray=True)
        TPP_traj = TPP.pred_net.pred_traj(TPP.pack_veh_surr(), time_len=T, to_ndarray=True)
        CPP_traj = CPP.pred_net.pred_traj(CPP.pack_veh_surr(), time_len=T, to_ndarray=True)

        assert len(TPP_traj) == len(TP_traj) == round(T / self.dt) + 1, f"{len(TPP_traj)} {len(TP_traj)} {T}"

        have_platoon = False
        have_weaving = False

        platoon_feas = (isinstance(CR, Game_A_Vehicle) and CR_stra.lc_direction != 0
                        and EV_stra.lc_direction == CR_stra.lc_direction)
        weaving_feas = (isinstance(TR, Game_A_Vehicle) and TR_stra.lc_direction != 0
                        and EV_stra.lc_direction == - TR_stra.lc_direction)
        if platoon_feas and weaving_feas:
            return None, None
        elif platoon_feas or weaving_feas:
            if platoon_feas:
                # print("platoon")
                ev_res_platoon, cr_res_platoon = opti_quintic_given_T_platoon(
                    lc_surr, T, TP_traj=TP_traj, TR_traj=TR_traj, CP_traj=CP_traj, CRR_traj=CRR_traj,
                    ori_lane=EV_stra.lane, target_lane=EV_stra.target_lane
                )
                if ev_res_platoon is None:
                    return None, None
                print("platoon solved")
                have_platoon = True

                EV_stra.solve_res = ev_res_platoon
                CR_stra.solve_res = cr_res_platoon
                CR_traj = cr_res_platoon.traj
            if not have_platoon and weaving_feas:
                print("weaving")
                ev_res_weaving, tr_res_weaving = opti_quintic_given_T_weaving(
                    lc_surr, T, TP_traj=TP_traj, TRR_traj=TRR_traj, CP_traj=CP_traj, CR_traj=CR_traj,
                    ori_lane=EV_stra.lane, target_lane=EV_stra.target_lane
                )
                if ev_res_weaving is None:
                    return None, None
                print("weaving solved")
                have_weaving = True

                EV_stra.solve_res = ev_res_weaving
                TR_stra.solve_res = tr_res_weaving
                TR_traj = tr_res_weaving.traj
        # elif EV_stra.lc_direction != 0:
        else:
            # print("single lc")
            ev_res_single = opti_quintic_given_T_single(
                lc_surr, T, TP_traj=TP_traj, TR_traj=TR_traj, CP_traj=CP_traj, CR_traj=CR_traj,
                ori_lane=EV_stra.lane, target_lane=EV_stra.target_lane
            )
            if ev_res_single is None:
                return None, None
            # print("single lc solved")
            EV_stra.solve_res = ev_res_single
        # else:
        #     # print("keep lane")
        #     EV_traj, _ = self.pred_self_traj(EV_stra, to_ndarray=True, ache=True)
        #     _, EV_real_cost, _ = cal_other_cost(
        #         self, self.rho, EV_traj, [CP_traj],
        #         v_length_s=[CP.length], stra_info=EV_stra
        #     )

        EV_traj = EV_stra.solve_res.traj

        TF_rho_hat = TP.rho if isinstance(TP, Game_A_Vehicle) else self.get_rho_hat(TP)
        TR_rho_hat = TR.rho if isinstance(TR, Game_A_Vehicle) else self.get_rho_hat(TR)
        CP_rho_hat = CP.rho if isinstance(CP, Game_A_Vehicle) else self.get_rho_hat(CP)
        CR_rho_hat = CR.rho if isinstance(CR, Game_A_Vehicle) else self.get_rho_hat(CR)

        TP_cost_hat, TP_real_cost, TP_cost_lambda = cal_other_cost(
            TP, TF_rho_hat, TP_traj, [TPP_traj],
            v_length_s=[TR.length], stra_info=TP_stra
        )
        CP_cost_hat, CP_real_cost, CP_cost_lambda = cal_other_cost(
            CP, CP_rho_hat, CP_traj, [CPP_traj],
            v_length_s=[CPP.length], stra_info=CP_stra
        )

        if lc_direction != 0:
            if have_weaving:
                TR_cost_hat = TR_real_cost = TR_stra.solve_res.cost
                TR_cost_lambda = None
            else:
                if have_platoon:
                    route_cost = CR_stra.solve_res.route + EV_stra.solve_res.route
                else:
                    route_cost = EV_stra.solve_res.route
                route_cost = 0 if TR_stra.lc_direction != 0 else route_cost
                route_cost = route_cost if route_cost < 0 else 0
                TR_cost_hat, TR_real_cost, TR_cost_lambda = cal_other_cost(
                    TR, TR_rho_hat, TR_traj,
                    [EV_traj, TP_traj, CR_traj],
                    v_length_s=[self.length, TP.length, CR.length],
                    route_cost=route_cost, stra_info=TR_stra
                )
            if have_platoon:
                CR_cost_hat = CR_real_cost = CR_stra.solve_res.cost
                CR_cost_lambda = None
            else:
                if have_weaving:
                    route_cost = TR_stra.solve_res.route
                else:
                    route_cost = 0
                route_cost = 0 if CR_stra.lc_direction != 0 else route_cost
                route_cost = route_cost if route_cost < 0 else 0
                CR_cost_hat, CR_real_cost, CR_cost_lambda = cal_other_cost(
                    CR, CR_rho_hat, CR_traj,
                    [EV_traj, CP_traj, TR_traj],
                    v_length_s=[self.length, CP.length, TR.length],
                    route_cost=route_cost, stra_info=CR_stra
                )
        else:
            CR_cost_hat, CR_real_cost, CR_cost_lambda = cal_other_cost(
                CR, CR_rho_hat, CR_traj,
                [EV_traj], v_length_s=[self.length], stra_info=CR_stra
            )
            TR_cost_hat, TR_real_cost, TR_cost_lambda = cal_other_cost(
                TR, TR_rho_hat, TR_traj,
                [TP_traj], v_length_s=[TP.length], stra_info=TR_stra
            )

        total_cost = EV_stra.solve_res.cost
        if isinstance(TR, Game_A_Vehicle):
            total_cost += TR.game_co * TR_real_cost
        if isinstance(CR, Game_A_Vehicle):
            total_cost += CR.game_co * CR_real_cost
        if isinstance(TP, Game_A_Vehicle):
            total_cost += TP.game_co * TP_real_cost
        if isinstance(CP, Game_A_Vehicle):
            total_cost += CP.game_co * CP_real_cost

        result = [
            T, EV_stra, TP_stra, TR_stra, CR_stra, CP_stra, total_cost,
            EV_stra.solve_res.cost, TP_cost_hat, TR_cost_hat, CR_cost_hat, CP_cost_hat,
            TP_real_cost, TR_real_cost, CR_real_cost, CP_real_cost
        ]

        traj_data = TrajData(
            EV_traj, TP_traj, TR_traj, CP_traj, CR_traj,
            TP_cost_lambda, TR_cost_lambda, CP_cost_lambda, CR_cost_lambda,
        )
        return result, traj_data

    def set_path_mpc(self, solve_res: SolveRes):
        # x, vx, ax, y, vy, ay
        ref_path_ori = solve_res.traj
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
        ref_path = ReferencePath(ref_path, self.dt)
        self.mpc_solver = MPC_Solver(self.N_MPC, ref_path, self)
        self.mpc_solver.init_mpc()
        return ref_path_ori


class Game_H_Vehicle(Game_Vehicle):
    NAME = "Game-HV"

    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.rho_hat_s = {}

    def get_strategies(self, game_surr: GameVehSurr, single_stra=False) -> list[StraInfo]:
        if single_stra:
            return [StraInfo(self, None, 1, 0)]  # None: 与EV时长保持一致，1：stra，0：不换道
        strategy = list(itertools.product([None], self.cf_stra_s, [0]))
        strategy = [StraInfo(self, stra_time, cf_stra, lc_direction)
                    for stra_time, cf_stra, lc_direction in strategy]
        return strategy

    def lc_intention_judge(self):
        if (self.opti_gap is not None and self.is_keep_lane_center()
                and self.opti_gap.target_lane == self.lane):
            self.reset_lc_state()
        super().lc_intention_judge()

    def cal_vehicle_control(self):
        if self.opti_gap is not None and self.opti_gap.target_lane != self.lane:
            if self.opti_gap.adapt_end_time > self.lane.time_:
                self.next_acc = self.opti_gap.target_acc
            else:
                self.target_lane = self.opti_gap.target_lane
                veh_surr = VehSurr(ev=self, cp=self.opti_gap.TF)
                # veh_surr = VehSurr(ev=self, cp=self.target_lane.get_relative_car(self)[0])
                self.next_acc = self.cf_model.step(veh_surr)
        else:
            next_acc_block = np.inf
            if self.lane.index not in self.destination_lane_indexes:
                next_acc_block = (
                    self.cf_model.step(VehSurr(ev=self, cp=self.lane.road.end_weaving_block_veh))
                )
            next_acc = self.cf_model.step(self.pack_veh_surr())
            self.next_acc = min(next_acc, next_acc_block)

        self.next_delta = self.cf_lateral_control()
        return self.next_acc, self.next_delta


class Game_O_Vehicle(Game_H_Vehicle):
    NAME = "Game-OV"

    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)

    def get_strategies(self, game_surr: GameVehSurr, single_stra=False) -> list[StraInfo]:
        return [StraInfo(self, None, 1, 0)]  # None: 与EV时长保持一致，1：stra，0：不换道

    def pred_self_traj(self, stra_info: StraInfo = None, to_ndarray=True, ache=True):
        time_len = stra_info.stra_time
        step_num = round(time_len / self.dt) + 1
        if to_ndarray:
            traj_list = [self.get_traj_point().to_ndarray()] * step_num
        else:
            traj_list = [self.get_traj_point()] * step_num
        return np.array(traj_list), None
