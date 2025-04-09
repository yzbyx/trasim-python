# -*- coding: utf-8 -*-
# @Time : 2025/3/22 22:13
# @Author : yzbyx
# @File : game_agent.py
# Software: PyCharm
import abc
import itertools
from abc import ABC
from typing import Optional, TYPE_CHECKING
from unittest.util import safe_repr

import cvxpy
import numpy as np
import pandas as pd
import sympy as sp

from trasim_simplified.core.agent.agent import AgentBase
from trasim_simplified.core.agent.collision_risk import calculate_max_collision_risk
from trasim_simplified.core.agent.fuzzy_logic import FuzzyLogic
from trasim_simplified.core.agent.mpc_solver import MPC_Solver
from trasim_simplified.core.agent.ref_path import ReferencePath
from trasim_simplified.core.agent.utils import get_xy_quintic, get_y_guess, get_x_guess
from trasim_simplified.core.constant import RouteType, VehSurr, TrajPoint

if TYPE_CHECKING:
    from trasim_simplified.core.agent import Vehicle
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class Game_Vehicle(AgentBase, ABC):
    def __init__(self, lane: 'LaneAbstract', type_: int, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.T_b = 1.5
        """临界期望时距"""
        self.rho = 0.5
        """激进系数（仅用于HV）"""

        self.lc_acc_gain = []
        """换道加速度增益"""

        self.T_safe = 0.5
        self.T_desire = 1
        self.Rho_ori = 0.5

        self.lc_risk = None
        self.lc_benefit = None
        self.aggressiveness = 0.5
        self.e_route_threshold = 0.5

        self.rho_hat_s = {}  # 储存车辆id_对应的rho_hat

    def cal_safe_cost(self, traj, other_traj, k_sf=1):
        """计算与其他轨迹的安全成本"""
        if other_traj is None:
            return -np.inf
        assert len(traj) == len(other_traj), f"The length of traj is not equal to the other_traj."
        x_f = traj[-1, 0]
        v_f = traj[-1, 1]
        x_other_f = other_traj[-1, 0]
        gap = x_other_f - x_f
        if gap < 0:
            gap = x_f - x_other_f
        return k_sf * (self.T_safe * v_f - (gap - 5))

    def _get_rho_hat_s(self, vehicles):
        rho_hat_s = []
        for v in vehicles:
            if v is not None:
                rho_hat_s.append(self.rho_hat_s.get(v.id_, 0.5))
            else:
                rho_hat_s.append(None)
        return rho_hat_s

    def set_game_stra(self, stra):
        """设置策略"""
        self.is_gaming = True
        self.game_time_wanted = stra

    def clear_game_stra(self):
        """清除策略"""
        self.is_gaming = False
        self.game_time_wanted = None

    def cal_cost_by_traj(self, traj, PC_traj=None, LC_traj=None, TR_traj=None, TF_traj=None,
                         rho=None, return_lambda=False, is_lc=False):
        """只有不换道车辆才会使用此方法
        :param traj: 预测轨迹 x, dx, ddx, y, dy, ddy
        :param PC_traj: 预测前车轨迹
        :param LC_traj: 预测换道车轨迹
        :param TR_traj: 换道车辆目标车道后车轨迹
        :param TF_traj: 换道车辆目标车道前车轨迹
        :param rho: 激进系数
        :param return_lambda: 是否返回lambda函数
        """
        k_c = 1
        k_e = 1
        k_sp = 1
        k_com = 0.5
        k_eff = 0.5

        # 计算安全成本
        safe_cost_lc = self.cal_safe_cost(traj, LC_traj)
        safe_cost_pc = self.cal_safe_cost(traj, PC_traj)
        safe_cost_tr = self.cal_safe_cost(traj, TR_traj)
        safe_cost_tf = self.cal_safe_cost(traj, TF_traj)

        safe_cost = max(safe_cost_lc, safe_cost_pc, safe_cost_tr, safe_cost_tf)
        # 舒适性计算 横纵向最大加速度、横摆角速度
        ddx_max = max(abs(traj[:, 2]))
        ddy_max = max(abs(traj[:, 5]))
        yaw = np.arctan2(np.diff(traj[:, 3]), np.diff(traj[:, 0]))
        d_yaw_max = max(abs(np.diff(yaw) / self.dt))

        com_cost = k_c * (ddx_max + ddy_max + d_yaw_max)  # 舒适性计算
        # # 效率计算 单位时间内的平均速度-初始速度
        speed = np.sqrt(traj[:, 1] ** 2 + traj[:, 4] ** 2)
        eff_cost = k_e * np.mean(speed) - speed[0]  # 效率计算

        def cost(rho_):
            total_cost = ((1 - rho_) * safe_cost + rho_ * (k_com * com_cost + k_eff * eff_cost))
            return total_cost

        if return_lambda:
            return cost

        return cost(rho)

    @abc.abstractmethod
    def get_strategies(self, is_lc: bool = False):
        """获取车辆策略"""
        pass

    @abc.abstractmethod
    def pred_self_traj(self, stra=None, T=None, uniform_speed=False,
                       is_game=False, target_lane: "LaneAbstract" = None,
                       PC_traj=None, time_len=None):
        """在策略下预测自车轨迹"""
        pass

    def reset_game(self, call_game_veh_id=None):
        self.rho = self.Rho_ori
        # TODO reset_game

    def pred_short_risk(self, time_len=3):
        pass

    def pred_risk(self, time_len=3):
        # 预测周边4车辆的3s轨迹
        traj_cp, traj_lp, traj_lr, traj_rp, traj_rr = [
            self.pred_net.pred_traj(veh.pack_veh_surr(), type_="const", time_len=3)
            if veh is not None else None
            for veh in [self.leader, self.lf, self.lr, self.rf, self.rr]
        ]

        if self.left_lane is not None:
            traj_ev_left = self.pred_self_traj(
                target_lane=self.left_lane, PC_traj=traj_lp, is_game=False, time_len=3
            )
            left_collision_prob = calculate_max_collision_risk(
                traj_ev_left, self.width, self.length,
                traj_lp, self.lf.width, self.lf.length,
                traj_lr, self.lf.width, self.lf.length,
                traj_cp, self.f.width, self.f.length
            )
        else:
            left_collision_prob = 1
        if self.right_lane is not None:
            traj_ev_right = self.pred_self_traj(
                target_lane=self.right_lane, PC_traj=traj_rp, is_game=False, time_len=3
            )
            right_collision_prob = calculate_max_collision_risk(
                traj_ev_right, self.width, self.length,
                traj_rp, self.rf.width, self.rf.length,
                traj_rr, self.rf.width, self.rf.length,
                traj_cp, self.f.width, self.f.length
            )
        else:
            right_collision_prob = 1
        return left_collision_prob, right_collision_prob  # TODO：可能要归一化

    def pred_benefit(self):
        target_lane, res = self.lc_model.step(self.pack_veh_surr())
        acc_gain = res["acc_gain"]  # left, right换道的加速度收益
        left_acc_gain = acc_gain[0]
        right_acc_gain = acc_gain[1]

        # 目标路径
        target_direction = 0
        if self.route_type == RouteType.diverge and self.lane.index not in self.destination_lane_indexes:
            current_lane_index = self.lane.index
            least_lane_index = min(self.destination_lane_indexes)
            n_need_lc = least_lane_index - current_lane_index
            d_r = self.lane.road.end_weaving_pos - self.x
            t_r = d_r / self.v
            e_r = max([0, 1 - d_r / (n_need_lc * 20), 1 - t_r / (n_need_lc * 3)])  # 预期20m一次换道，5s一次换道
            target_direction = 1
        elif self.route_type == RouteType.merge and self.lane.index not in self.destination_lane_indexes:
            current_lane_index = self.lane.index
            max_lane_index = max(self.destination_lane_indexes)
            n_need_lc = current_lane_index - max_lane_index
            d_r = self.x - self.lane.road.start_weaving_pos
            t_r = d_r / self.v
            e_r = max([0, 1 - d_r / (n_need_lc * 20), 1 - t_r / (n_need_lc * 3)])
            target_direction = -1
        else:
            e_r = 0

        # 计算换道收益
        if e_r >= self.e_route_threshold:
            if target_direction == 1:
                left_gain = 0
                right_gain = right_acc_gain + e_r
            else:
                left_gain = left_acc_gain + e_r
                right_gain = 0
        elif e_r < self.e_route_threshold and target_direction != 0:
            if target_direction == 1:
                left_gain = left_acc_gain
                right_gain = right_acc_gain + e_r
            else:
                left_gain = left_acc_gain + e_r
                right_gain = right_acc_gain
        else:
            left_gain = left_acc_gain
            right_gain = right_acc_gain

        return left_gain, right_gain  # TODO：可能要归一化

    def lc_intention_judge(self):
        # 处于跟驰状态或有换道中有碰撞风险，重新计算目标车道
        if not self.lane_changing or self.pred_short_risk():
            # 计算换道收益
            left_gain, right_gain = self.pred_benefit()
            # 计算换道风险
            left_risk, right_risk = self.pred_risk()
            if left_gain >= right_gain:
                self.lc_direction = 1
                self.lc_benefit = left_gain
                self.lc_risk = left_risk
            else:
                self.lc_direction = -1
                self.lc_benefit = right_gain
                self.lc_risk = right_risk


class Game_A_Vehicle(Game_Vehicle):
    """自动驾驶车辆"""

    def __init__(self, lane: 'LaneAbstract', type_: int, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.TF_co = 1  # TF自动驾驶的公平协作程度 [0, 1]
        self.combine_vehicles: list[Game_Vehicle] = []
        """TR, TF, PC"""
        self.cost_df = None  # 存储博弈策略与对应效用值
        self.opti_idx = None  # 存储最优策略的索引

        self.LC_pred_traj_df: Optional[pd.DataFrame] = None
        """换道车对该车轨迹的预测"""

        self.N_MPC = 20
        self.mpc_solver: Optional[MPC_Solver] = None

        self.lc_step_length = -1  # 换道步数
        self.lc_end_step = -1  # 换道完成时的仿真步数
        self.lc_start_step = -1  # 换道开始时的仿真步数
        self.lc_entered_step = -1  # 车辆中心进入目标车道的仿真步数
        self.lc_step_conti = 0  # 如果正在换道，持续的换道步数
        self.lc_start_state = None  # 换道开始点状态[x, dx, ddx, y, dy, ddy]
        self.lc_end_state = None  # 换道结束点目标状态[x, dx, ddx, y, dy, ddy]
        self.lc_x_opti = None  # 换道轨迹参数

        self.lc_time_max = 10  # 换道最大时间
        self.lc_time_min = 1  # 换道最小时间

    def lc_intention_judge(self):
        # 处于跟驰状态或有换道中有碰撞风险，重新计算目标车道
        super().lc_intention_judge()

    def lc_decision_making(self):
        """判断是否换道"""
        # 根据上一时间步的TR行为策略更新rho_hat
        self.update_rho_hat()
        self.state = self.get_state_for_traj()
        if self.lane_changing or self.lc_direction != 0:
            # LC轨迹优化，如果博弈选择不换道，需要重新更新self.lane_changing以及self.target_lane
            self.stackel_berg()
            # 计算参考轨迹
            if self.lc_direction != 0:
                self.cal_ref_path()

    def cal_vehicle_control(self):
        """横向控制"""
        if self.lane.step_ >= self.lc_end_step:
            self.lane_changing = False

            if self.lane.step_ == self.lc_end_step:
                # 恢复TR、TF的TIME_WANTED
                TR = self.combine_vehicles[0]
                TR.reset_game(self.ID)
                TF = self.combine_vehicles[1]
                TF.reset_game(self.ID)

            self.lc_step_conti = 0
            self.lc_direction = 0
            self.target_lane = self.lane

            delta = self.cf_lateral_control()
            acc = self.cf_model.step(self.pack_veh_surr())
        else:
            self.lane_changing = True
            # 横向控制
            acc, delta = self.mpc_solver.step_mpc()  # 默认计算出的符合约束的加速度和转向角
            self.lc_step_conti += 1

        self.next_acc = acc
        self.next_delta = delta

    def update_rho_hat(self):
        """求解rho的范围，入股TR的rho_hat不在范围内，更新TR的rho_hat"""
        if self.combine_vehicles is None or (self.lane_changing and self.road.step_ >= self.lc_entered_step):
            return
        if self.cost_df is None:
            return
        if len(self.combine_vehicles) == 0:
            return
        TR = self.combine_vehicles[0]
        if isinstance(TR, Game_H_Vehicle):
            if self.rho_hat_s.get(TR.ID, None) is None:
                # 第一次估计TR的rho_hat默认为0.5
                self.rho_hat_s[TR.ID] = 0.5
            TR_real_stra = TR.TIME_WANTED
            ego_stra = self.cost_df.loc[self.opti_idx]["ego_stra"]
            T = np.arange(0, ego_stra + self.dt / 2, self.dt)
            TR_traj_real, _ = TR.pred_self_traj(TR_real_stra, T, is_lc=False)

            TR_traj = self.LC_pred_traj_df.loc[self.opti_idx]["TR_traj"]
            TF_traj = self.LC_pred_traj_df.loc[self.opti_idx]["TF_traj"]
            ego_traj = self.LC_pred_traj_df.loc[self.opti_idx]["ego_traj"]
            TR_cost_lambda_pred = TR.cal_cost_by_traj(
                TR_traj, PC_traj=TF_traj, LC_traj=ego_traj, return_lambda=True)
            TR_cost_lambda_real = TR.cal_cost_by_traj(
                TR_traj_real, PC_traj=TF_traj, LC_traj=ego_traj, return_lambda=True)

            # 构建不等式求解问题
            rho_ = sp.symbols("rho_")
            # 定义不等式
            inequality = sp.Ge(TR_cost_lambda_pred(rho_), TR_cost_lambda_real(rho_))  # f(x) >= g(x)
            # 求解不等式
            solution = sp.solve_univariate_inequality(inequality, rho_, relational=False)
            # 分析解集
            if isinstance(solution, sp.Union):
                intervals = solution.args
            else:
                intervals = [solution]

            for interval in intervals:
                if isinstance(interval, sp.Interval):
                    # 获取区间的上下限
                    lower_bound = max(0, interval.start)
                    upper_bound = min(1, interval.end)
                    # 更新TR的rho_hat
                    if lower_bound <= self.rho_hat_s[TR.ID] <= upper_bound:
                        continue
                    else:
                        self.rho_hat_s[TR.ID] = (lower_bound + upper_bound) / 2

    def cal_ref_path(self):
        times = np.arange(0, (self.lc_step_length + 0.5) * self.dt, self.dt)
        # x, dx, ddx, y, dy, ddy
        ref_path = np.vstack([np.array(get_xy_quintic(self.lc_x_opti, t)) for t in times])
        x, dx, ddx, y, dy, ddy = list(self.lc_end_state)
        times_after = np.arange(self.dt, (self.N_MPC + 0.5) * self.dt, self.dt).reshape(-1, 1)
        ref_path = np.vstack(
            [
                ref_path,
                np.hstack([
                    x + dx * times_after + 0.5 * ddx * times_after ** 2,
                    np.tile(dx, (self.N_MPC, 1)),
                    np.tile(ddx, (self.N_MPC, 1)),
                    np.tile(y, (self.N_MPC, 1)),
                    np.tile(dy, (self.N_MPC, 1)),
                    np.tile(ddy, (self.N_MPC, 1))
                ])
            ]
        )  # 填补参考轨迹
        ref_path = ReferencePath(ref_path, self.dt)
        self.mpc_solver = MPC_Solver(self.N_MPC, ref_path, self)
        self.mpc_solver.init_mpc()

    def stackel_berg(self):
        """基于stackelberg主从博弈理论的换道策略，计算得到参考轨迹（无论是否换道）"""
        # 计算不同策略（换道时间）的最优轨迹和对应效用值
        # cal_game_matrix需要更新周边车辆的策略，stackel_berg函数只更新自身的策略
        opti_df, TF, TR, PC = self.cal_game_matrix()
        # a0~a5, b0~b5, step, y1, ego_stra, TF_stra, TR_stra, PC_stra, ego_cost, TF_cost, TR_cost, PC_cost
        if len(opti_df) != 0:
            if isinstance(opti_df, pd.DataFrame):
                opti_df = opti_df[0]
            # 选择效用值最小的策略
            x_opt = opti_df.iloc[: 12].to_numpy().astype(float)
            T_opt = opti_df.iloc[14]
            step_opt = int(opti_df.iloc[12])

            self.lc_step_length = np.round(T_opt / self.dt).astype(int)
            self.lc_start_step = self.lane.step_
            self.lc_end_step = self.lc_start_step + self.lc_step_length
            self.lc_entered_step = self.lc_start_step + step_opt
            self.lc_end_state = get_xy_quintic(x_opt, T_opt)
            self.lc_start_state = self.get_state_for_traj()
            self.lc_x_opti = x_opt
        else:
            self.lc_direction = 0
            self.target_lane = self.lane

    def get_y_constraint(self, is_lc):
        lane_center_y = self.lane.y_center
        lane_width = self.lane.width

        if is_lc:
            y1 = lane_center_y + (- self.lc_direction) * lane_width
            y_limit_low = min(lane_center_y, y1) + self.width / 2
            y_limit_up = max(lane_center_y, y1) - self.width / 2
        else:
            y1 = lane_center_y
            y_limit_low = lane_center_y + lane_width / 2
            y_limit_up = lane_center_y - lane_width / 2

        return y1, y_limit_low, y_limit_up

    def cal_game_matrix(self):
        """backward induction method
        :return: opti_df (a0~a5, b0~b5, ego_stra, TF_stra, TR_stra, PC_stra,
         ego_cost, TF_cost, TR_cost, PC_cost)
        """
        TR, TF, PC = (self.lr, self.lf, self.f) if self.lc_direction == -1 else (self.rr, self.rf, self.f)
        try:
            if isinstance(TR, Game_H_Vehicle) and isinstance(TF, Game_H_Vehicle):
                # ego与TR的博弈
                TR_strategy = TR.get_strategies(is_lc=False)
                ego_strategy = self.get_strategies(is_lc=True)

                cost_df = self._cal_cost_df(
                    stra_product=itertools.product(ego_strategy, [None], TR_strategy, [None]),
                    cal_TF_cost=False, cal_TR_cost=True, cal_PC_cost=False,
                    TF=TF, TR=TR, PC=PC
                )

                # 根据估计的TR成本函数计算最小的当前车辆cost
                cost_df_temp = cost_df[cost_df["ego_cost"] != np.inf]
                min_TR_cost_idx = cost_df_temp.groupby('ego_stra')['TR_cost'].idxmin()  # 找到最小的TR_cost值
                min_cost_idx = cost_df.loc[min_TR_cost_idx]["ego_cost"].idxmin()  # 找到最小的cost值

                # 根据真实的TR成本函数计算真正的TR_stra（站在TR的立场上）
                cost_df_tr = cost_df[cost_df["TR_2_EV_pred_cost"] != np.inf]
                min_TR_cost_idx_ = cost_df_tr.groupby('ego_stra')['TR_real_cost'].idxmin()  # 找到最小的TR_cost值
                min_cost_idx_ = cost_df.loc[min_TR_cost_idx_]["TR_2_EV_pred_cost"].idxmin()
                temp_df = cost_df[(cost_df["ego_stra"] == cost_df.loc[min_cost_idx_]["ego_stra"])]
                min_TR_real_cost_idx = temp_df["TR_real_cost"].idxmin()

                # 更新TR的策略
                TIME_WANTED = temp_df.loc[min_TR_real_cost_idx]["TR_stra"]
                TR.set_game_stra(TIME_WANTED)
            elif isinstance(TR, Game_H_Vehicle) and isinstance(TF, Game_A_Vehicle):
                # ego与TR的博弈，ego与TF的合作
                TR_strategy = TR.get_strategies(is_lc=False)
                TF_strategy = TF.get_strategies(is_lc=False)
                ego_strategy = self.get_strategies(is_lc=True)

                cost_df = self._cal_cost_df(
                    stra_product=itertools.product(ego_strategy, TF_strategy, TR_strategy, [None]),
                    cal_TF_cost=True, cal_TR_cost=True, cal_PC_cost=False,
                    TF=TF, TR=TR, PC=PC
                )

                cost_df["total_cost"] = cost_df["ego_cost"] + TF.TF_co * cost_df["TF_cost"]  # 合作效用
                cost_df_temp = cost_df[cost_df["total_cost"] != np.inf]
                min_TR_cost_idx = cost_df_temp.groupby(by=["ego_stra", "TF_stra"])['total_cost'].idxmin()  # 找到最小的TR_cost值
                min_cost_idx = cost_df.loc[min_TR_cost_idx]["total_cost"].idxmin()  # 找到最小的cost值

                # 根据真实的TR成本函数计算真正的TR_stra（站在TR的立场上）
                TF_co_pred = TR.get_co_hat_s([TF])[0]
                cost_df["total_cost_TR_pred"] = (cost_df["TR_2_EV_pred_cost"] +
                                                 TF_co_pred * cost_df["TF_2_EV_pred_cost"])
                cost_df_temp = cost_df[cost_df["total_cost_TR_pred"] != np.inf]
                min_TR_cost_idx_ = cost_df_temp.groupby(by=["ego_stra", "TF_stra"])['total_cost_TR_pred'].idxmin()  # 找到最小的TR_cost值
                min_cost_idx_ = cost_df.loc[min_TR_cost_idx_]["total_cost_TR_pred"].idxmin()

                temp_df = cost_df[(cost_df["ego_stra"] == cost_df.loc[min_cost_idx_]["ego_stra"]) &
                                  (cost_df["TF_stra"] == cost_df.loc[min_cost_idx_]["TF_stra"]) &
                                  (cost_df["total_cost_TR_pred"] != np.inf)]
                min_TR_real_cost_idx = temp_df["TR_real_cost"].idxmin()

                # 更新TR的策略
                TIME_WANTED = temp_df.loc[min_TR_real_cost_idx]["TR_stra"]
                TR.set_game_stra(TIME_WANTED)
            elif isinstance(TR, Game_A_Vehicle) and isinstance(TF, Game_H_Vehicle):
                # ego与TR的合作
                TR_strategy = TR.get_strategies(is_lc=False)
                ego_strategy = self.get_strategies(is_lc=True)

                cost_df = self._cal_cost_df(
                    stra_product=itertools.product(ego_strategy, [None], TR_strategy, [None]),
                    cal_TF_cost=False, cal_TR_cost=True, cal_PC_cost=False,
                    TF=TF, TR=TR, PC=PC
                )

                cost_df["total_cost"] = cost_df["cost"] + TR.TF_co * cost_df["TR_cost"]  # 合作效用
                cost_df_temp = cost_df[cost_df["total_cost"] != np.inf]
                min_cost_idx = cost_df_temp["total_cost"].idxmin()  # 找到最小的cost值

                # 更新TR的策略
                TR.TIME_WANTED = cost_df.loc[min_cost_idx]["TR_stra"]
            elif isinstance(TR, Game_A_Vehicle) and isinstance(TF, Game_A_Vehicle):
                # ego与TR的合作，ego与TF的合作
                TR_strategy = TR.get_strategies(is_lc=False)
                TF_strategy = TF.get_strategies(is_lc=False)
                ego_strategy = self.get_strategies(is_lc=True)

                cost_df = self._cal_cost_df(
                    stra_product=itertools.product(ego_strategy, TF_strategy, TR_strategy, [None]),
                    cal_TF_cost=True, cal_TR_cost=True, cal_PC_cost=False,
                    TF=TF, TR=TR, PC=PC
                )

                cost_df["total_cost"] = cost_df["cost"] + TR.TF_co * cost_df["TR_cost"] + TF.TF_co * cost_df["TF_cost"]
                cost_df_temp = cost_df[cost_df["total_cost"] != np.inf]
                min_cost_idx = cost_df_temp["total_cost"].idxmin()  # 找到最小的cost值

                TR.TIME_WANTED = cost_df.loc[min_cost_idx]["TR_stra"]
                TF.TIME_WANTED = cost_df.loc[min_cost_idx]["TF_stra"]
            else:
                raise NotImplementedError("未知的车辆类型")
        except cvxpy.error.SolverError:
            return [], TF, TR, PC
        self.cost_df = cost_df
        self.opti_idx = min_cost_idx

        return cost_df.loc[min_cost_idx], TF, TR, PC

    def _cal_cost_df(self, stra_product, cal_TF_cost, cal_TR_cost, cal_PC_cost, TF, TR, PC):
        stra_product = list(stra_product)
        num = len(stra_product)
        self.LC_pred_traj_df = pd.DataFrame(
            data=np.zeros((num, 8), dtype=object),
            columns=[*[f"{name}_{type_}" for type_ in ["stra", "traj"]
                       for name in ["ego", "TF", "TR", "PC"]]]
        )
        df = pd.DataFrame(
            np.vstack(
                list(
                    list(self.cal_cost(
                        ego_stra, TF=TF, TR=TR, PC=PC,
                        TF_stra=TF_stra, TR_stra=TR_stra, PC_stra=PC_stra,
                        cal_TF_cost=cal_TF_cost, cal_TR_cost=cal_TR_cost, cal_PC_cost=cal_PC_cost, index=i
                    )) for i, [ego_stra, TF_stra, TR_stra, PC_stra] in enumerate(stra_product)
                )
            ),
            columns=[*[f"{i}{j}" for i in ["a", "b"] for j in range(6)], *["step", "y1"],
                     *[f"{name}_{type_}" for type_ in ["stra", "cost", "real_cost"]
                       for name in ["ego", "TF", "TR", "PC"]]] + ["TR_2_EV_pred_cost", "TR_2_TF_pred_cost"]
        )
        return df

    def cal_cost(self, stra,
                 TF: Game_Vehicle = None, TR: Game_Vehicle = None, PC: Game_Vehicle = None,
                 TF_stra=None, TR_stra=None, PC_stra=None,
                 cal_TF_cost=False, cal_TR_cost=False, cal_PC_cost=False, index=None):
        """根据策略计算效用值
        :return: a0~a5, b0~b5, step, y1, T, TF_stra, TR_stra, PC_stra, cost, TF_cost, TR_cost, PC_cost
        """
        # 换道效用
        y1, y_limit_low, y_limit_up = self.get_y_constraint(is_lc=True)
        lc_result, [TF_traj, TR_traj, PC_traj, TF_PC_traj, TR_PC_traj, PC_PC_traj] = \
            self.opti_quintic_given_T(
                stra, y1, y_limit_low, y_limit_up,
                TF, TR, PC,
                TF_stra, TR_stra, PC_stra,
                return_other_traj=True
            )
        if lc_result[0] == np.inf:
            return np.hstack([lc_result[1:], [TF_stra, TR_stra, PC_stra],
                              [lc_result[0], np.nan, np.nan, np.nan,
                               np.nan, np.nan, np.nan, np.nan]]).reshape(1, -1)

        time_steps = np.arange(0, stra + self.dt / 2, self.dt)
        cost, traj = lc_result[0], np.vstack(
            [np.array(get_xy_quintic(lc_result[1:13], t)) for t in time_steps]
        )

        self.LC_pred_traj_df.loc[index, 'ego_stra'] = stra
        self.LC_pred_traj_df.loc[index, 'TF_stra'] = TF_stra
        self.LC_pred_traj_df.loc[index, 'TR_stra'] = TR_stra
        self.LC_pred_traj_df.loc[index, 'PC_stra'] = PC_stra

        self.LC_pred_traj_df.loc[index, 'ego_traj'] = traj
        self.LC_pred_traj_df.loc[index, 'TF_traj'] = TF_traj
        self.LC_pred_traj_df.loc[index, 'TR_traj'] = TR_traj
        self.LC_pred_traj_df.loc[index, 'PC_traj'] = PC_traj

        TF_rho, TR_rho, PC_rho = self._get_rho_hat_s([TF, TR, PC])

        # TF
        TF_cost = TF.cal_cost_by_traj(
            TF_traj, PC_traj=TF_PC_traj, rho=TF_rho) if cal_TF_cost else np.nan
        TF_real_cost = TF_cost if isinstance(TF, Game_A_Vehicle) else TF.cal_cost_by_traj(
            TF_traj, PC_traj=TF_PC_traj, rho=TF.rho) if cal_TF_cost else np.nan

        # TR
        TR_cost = TR.cal_cost_by_traj(
            TR_traj, PC_traj=TR_PC_traj, LC_traj=traj, rho=TR_rho) if cal_TR_cost else np.nan
        TR_real_cost = TR_cost if isinstance(TR, Game_A_Vehicle) else (TR.cal_cost_by_traj(
            TR_traj, PC_traj=TR_PC_traj, LC_traj=traj, rho=TR.rho) if cal_TR_cost else np.nan)

        # PC
        PC_cost = PC.cal_cost_by_traj(
            PC_traj, PC_traj=PC_PC_traj, rho=PC_rho) if cal_PC_cost else np.nan
        PC_real_cost = PC_cost if isinstance(PC, Game_A_Vehicle) else PC.cal_cost_by_traj(
            PC_traj, PC_traj=PC_PC_traj, rho=PC.rho) if cal_PC_cost else np.nan

        # TR对EV的效用估计
        TR_2_EV_pred_cost = np.nan  # TR对EV的效用估计
        TR_2_TF_pred_cost = np.nan  # TR对TF的效用估计
        if TR is not None and isinstance(TR, Game_H_Vehicle):
            EV_rho_hat = TR._get_rho_hat_s([self])[0]
            TR_2_EV_pred_cost = self.cal_cost_by_traj(
                traj, PC_traj=PC_traj, TR_traj=TR_traj, TF_traj=TF_traj, rho=EV_rho_hat
            )
            if TF is not None and isinstance(TR, Game_A_Vehicle):
                TF_rho_hat = TR._get_rho_hat_s([TF])[0]
                TR_2_TF_pred_cost = TR.cal_cost_by_traj(
                    TR_traj, PC_traj=TR_PC_traj, rho=TF_rho_hat)

        result = np.hstack(
            [lc_result[1:], [TF_stra, TR_stra, PC_stra],
             [cost, TF_cost, TR_cost, PC_cost,
              np.nan, TF_real_cost, TR_real_cost, PC_real_cost, TR_2_EV_pred_cost, TR_2_TF_pred_cost]]
        ).reshape(1, -1)
        return result

    def pred_self_traj(self, stra, T, is_lc: bool = False, self_uniform_speed=False):
        if is_lc:
            raise NotImplementedError("换道车辆不使用此方法")
        else:
            if self_uniform_speed:
                traj = np.array([
                    [self.position[0]] + [self.position[0] + self.speed * t for t in T],
                    [self.velocity[0]] * (len(T) + 1),
                    [0] * (len(T) + 1)
                ]).T
                PC_traj = None
            else:
                ori_T_want = self.TIME_WANTED
                self.TIME_WANTED = stra if stra is not None else ori_T_want
                # 假设前车匀速行驶
                traj, PC_traj = self.pred_self_traj(self.f, T, PC_traj=None)
                self.TIME_WANTED = ori_T_want
        return traj, PC_traj

    def get_strategies(self, is_lc: bool = False):
        if is_lc:
            # 换道时间
            strategy = list(
                np.arange(max(self.dt, self.lc_time_min),
                          max(self.dt, self.lc_time_max), self.dt * 10)
            )  # ATTENTION: 10倍步长
        else:
            # 期望时距
            strategy = list(np.arange(0.1, 5.1, 0.5))
        return strategy

    def opti_quintic_given_T(self, T, y1, y_limit_low, y_limit_up,
                             TF: Game_Vehicle = None, TR: Game_Vehicle = None, PC: Game_Vehicle = None,
                             TF_stra=None, TR_stra=None, PC_stra=None, return_other_traj=False):
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
        x = cvxpy.Variable(12 + 3)  # fxt，fyt的五次多项式系数 + 安全松弛变量

        A2 = np.array([
            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],  # y1
            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],  # dy1
            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]  # ddy1
        ])
        b2 = np.array([y1, 0, 0])

        constraints += [A @ x[: 12] == self.state]  # 初始状态约束
        constraints += [A2 @ x[6: 12] == b2]  # 终止状态约束

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
        b_ineq3 = np.array([0] * len(A_ineq3))
        b_ineq4 = np.array([self.V_MAX] * len(A_ineq3))
        constraints += [A_ineq3 @ x[: 6] >= b_ineq3]
        constraints += [A_ineq3 @ x[: 6] <= b_ineq4]

        # 计算每个时间步的ddx和ddy约束
        A_ineq5 = np.array([
            [0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3] for t in time_steps
        ])
        b_ineq5 = np.array([self.A_MAX] * len(A_ineq5))
        b_ineq6 = np.array([self.A_MAX] * len(A_ineq5))
        constraints += [A_ineq5 @ x[6: 12] <= b_ineq5]
        constraints += [A_ineq5 @ x[6: 12] >= b_ineq6]
        constraints += [A_ineq5 @ x[: 6] <= b_ineq5]
        constraints += [A_ineq5 @ x[: 6] >= b_ineq6]

        # 计算换道关键时间点的安全约束
        # 预测换道结束时目标车道前后车辆位置
        vehicles = [TR, TF, PC] if len(self.combine_vehicles) == 0 else self.combine_vehicles
        ([TF_traj, l_TF, TF_PC_traj],
         [TR_traj, l_TR, TR_PC_traj],  # FIXME: 此处的TR预测轨迹假设不受ego的换道影响
         [PC_traj, l_PC, PC_PC_traj]) = (
            self.pred_traj(
                time_steps,
                vehicles,
                stra_s=[TF_stra, TR_stra, PC_stra]
            )
        )
        xt_TF, dxt_TF = TF_traj[:, 0], TF_traj[:, 1]
        xt_TR, dxt_TR = TR_traj[:, 0], TR_traj[:, 1]
        xt_PC, dxt_PC = PC_traj[:, 0], PC_traj[:, 1]

        # 提取换道至目标车道初始时刻
        y, yf, dyf, ddyf = get_y_guess(T, self.state, y1)
        A_y = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])
        Y = A_y @ y
        # 进入目标车道的初始时刻 # ATTENTION：由于为车头的中点的y坐标，因此初始时刻为近似
        step = np.where((np.diff((Y <= (y_limit_low + y_limit_up) / 2)) != 0))[0][0] + 1
        lc_before_times = time_steps[:step]
        lc_after_times = time_steps[step:]
        A_ineq_7 = np.array([
            [1, t, t ** 2, t ** 3, t ** 4, t ** 5, 0, 0, 0, 0, 0, 0] for t in [lc_before_times[-1]]
        ])
        A_ineq_8 = np.array([
            [1, t, t ** 2, t ** 3, t ** 4, t ** 5, 0, 0, 0, 0, 0, 0] for t in [lc_after_times[-1]]
        ])
        # PC
        dxt_before = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in [lc_before_times[-1]]])
        v = dxt_before @ x[:6]
        rear_x = xt_PC[step] - l_PC
        b_ineq_7_part1 = rear_x - v * self.T_safe
        b_ineq_7_part2 = rear_x - v * self.T_desire
        constraints += [A_ineq_7 @ x[: 12] <= b_ineq_7_part1]
        constraints += [A_ineq_7 @ x[: 12] + x[12] == b_ineq_7_part2]
        # TF
        dxt_after = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in [lc_after_times[-1]]])
        v = dxt_after @ x[:6]
        rear_x = xt_TF[-1] - l_TF
        b_ineq_8_part1 = rear_x - v * self.T_safe
        b_ineq_8_part2 = rear_x - v * self.T_desire
        constraints += [A_ineq_8 @ x[: 12] <= b_ineq_8_part1]
        constraints += [A_ineq_8 @ x[: 12] + x[13] == b_ineq_8_part2]
        # TR
        rear_x = xt_TR[-1] + self.length
        b_ineq_9_part1 = rear_x + self.T_safe * dxt_TR[-1]
        b_ineq_9_part2 = rear_x + self.T_desire * dxt_TR[-1]
        constraints += [A_ineq_8 @ x[: 12] >= b_ineq_9_part1]
        constraints += [A_ineq_8 @ x[: 12] + x[14] == b_ineq_9_part2]

        # 安全性
        cost = cvxpy.abs(0)
        k_TP, k_TF, k_PC = 2, 2, 2
        cost += cvxpy.quad_form(x[12: 15], np.diag([k_TP, k_TF, k_PC]), assume_PSD=True)

        # 舒适性：横纵向jerk目标
        k_c = 10
        A_x_jerk = np.array([
            [0, 0, 0, 6, 24 * t, 60 * t ** 2, 0, 0, 0, 0, 0, 0] for t in time_steps
        ])
        # assert np.all(np.linalg.eigvals(A_x_jerk.T @ A_x_jerk) >= 0), "Matrix P is not positive semi-definite."
        cost += k_c * cvxpy.quad_form(x[:12], A_x_jerk.T @ A_x_jerk, assume_PSD=True) * self.dt

        # 效率
        k_e = 0.4
        A_dx = np.array([
            [0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4, 0, 0, 0, 0, 0, 0] for t in time_steps
        ])
        # cost += k_e * ((A_dx @ x[:12] - self.target_speed).T @ (A_dx @ x[:12] - self.target_speed)) * self.dt
        # 如果上式不行，可以用下面的
        target_speed_array = np.tile(self.target_speed, (len(time_steps), 1))
        cost += k_e * (cvxpy.quad_form(x[:12], A_dx.T @ A_dx, assume_PSD=True)
                       - x[:12].T @ A_dx.T @ target_speed_array
                       - target_speed_array.T @ A_dx @ x[:12]) * self.dt

        result = np.hstack([[np.inf], np.tile(np.nan, 12), [np.nan, y1, T]])
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        try:
            cost_value = prob.solve(verbose=False, solver=cvxpy.GUROBI)
            if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
                x_opt = x.value
                result = np.hstack([[cost_value], x_opt[:12], [step, y1, T]])
        except cvxpy.error.SolverError:
            pass

        if return_other_traj:
            other_traj = [TF_traj, TR_traj, PC_traj, TF_PC_traj, TR_PC_traj, PC_PC_traj]
            return result, other_traj
        return result

    def pred_traj(self, T, vehicles: list[Game_Vehicle], stra_s) -> list[tuple[np.ndarray, float, np.ndarray]]:
        # TODO pred_traj
        pred_traj_s = []
        for v, stra in zip(vehicles, stra_s):
            v = v if v is not None else self.f if self.lc_direction == 1 else self.r
            l = v.length
            traj, PC_traj = v.pred_self_traj(stra, T, is_lc=False, uniform_speed=False)
            pred_traj_s.append((traj, l, PC_traj))
        return pred_traj_s


class Game_H_Vehicle(Game_Vehicle):
    fuzzy_logic = FuzzyLogic()

    def __init__(self, lane: 'LaneAbstract', type_: int, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.co_default = 1  # 默认协作程度
        self.co_s = {}  # 储存车辆id_对应的协作程度
        self.combine_vehicles: list[Game_Vehicle] = []
        """TR, TF, PC"""
        self.cost_lambda = None

    def lc_decision_making(self):
        lc_prob = self.fuzzy_logic.compute(self.lc_benefit, self.lc_risk, self.aggressiveness)
        if self.lc_risk >= 0.9:
            return
        if self.lc_benefit >= 0.9:
            decision = True
        else:
            decision = 0.1 < lc_prob and lc_prob > np.random.uniform()
        if decision:
            self.lane_changing = True
            if self.lc_direction == 1:
                self.target_lane = self.right_lane
            elif self.lc_direction == -1:
                self.target_lane = self.left_lane

    def get_co_hat_s(self, vehicles: list[Game_Vehicle]):
        """获取协作程度"""
        co_s = []
        for v in vehicles:
            if isinstance(v, Game_A_Vehicle):
                co_s.append(self.co_s.get(self.ID, self.co_default))
            else:
                co_s.append(None)
        return co_s

    def get_strategies(self, is_lc: bool = False):
        if is_lc:
            # 换道方向
            strategy = [-1, 1]
        else:
            # 期望时距
            strategy = list(np.arange(0.1, 5.1, 0.5))
        return strategy

    def pred_self_traj(self, stra=None, T=None, uniform_speed=False,
                       is_game=False, target_lane: "LaneAbstract" = None,
                       PC_traj: list[TrajPoint] = None, time_len=None):
        if not is_game:
            traj_list = []
            # 估计轨迹
            veh = self.clone()
            veh.target_lane = target_lane
            veh.lane_changing = True
            leader = veh.clone()
            for step in range(round(time_len / self.dt)):
                if PC_traj is None:
                    leader.x = veh.x + 1000
                    leader.speed = veh.speed
                    leader.acc = 0
                else:
                    leader.x = PC_traj[step].x
                    leader.speed = PC_traj[step].speed
                    leader.acc = PC_traj[step].acc
                delta = veh.cf_lateral_control()
                acc = veh.cf_model.step(VehSurr(ev=veh, cp=leader))
                # print(acc, delta)
                # assert ~np.isnan(veh.x)
                veh.update_state(acc, delta)
                # assert ~np.isnan(veh.x)
                traj_list.append(veh.get_traj_point())
            return np.array(traj_list)

        if uniform_speed:
            traj = np.array([
                [self.position[0]] + [self.position[0] + self.speed * t for t in T],
                [self.velocity[0]] * (len(T) + 1),
                [0] * (len(T) + 1)
            ]).T
            PC_traj = None
        else:
            ori_T_want = self.TIME_WANTED
            self.TIME_WANTED = stra if stra is not None else ori_T_want
            # 假设前车匀速行驶
            traj, PC_traj = self.pred_self_traj(self.f, T, PC_traj=None)
            self.TIME_WANTED = ori_T_want
        return traj, PC_traj


class Game_O_Vehicle(Game_H_Vehicle):
    def cal_cost_by_traj(self, traj, PC_traj=None, LC_traj=None, rho=None, return_lambda=False):
        def cost():
            return 0

        if return_lambda:
            return cost
        return cost()

    def get_strategies(self, is_lc: bool = False):
        return [self.TIME_WANTED]

    def pred_self_traj(self, stra, T, is_lc: bool = False, uniform_speed=False):
        if np.isscalar(T):
            T = np.array([T])
        return (np.hstack(
            [np.tile(self.position[0], len(T)).reshape(-1, 1),
             np.zeros(len(T)).reshape(-1, 1),
             np.zeros(len(T)).reshape(-1, 1)]
        ).reshape(-1, 3), None)

    def lane_change_judge(self):
        pass
