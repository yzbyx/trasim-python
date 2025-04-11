# -*- coding: utf-8 -*-
# @Time : 2025/3/22 22:13
# @Author : yzbyx
# @File : game_agent.py
# Software: PyCharm
import abc
import itertools
from abc import ABC
from typing import Optional, TYPE_CHECKING

import cvxpy
import numpy as np
import pandas as pd
import sympy as sp

from trasim_simplified.core.agent.agent import AgentBase
from trasim_simplified.core.agent.collision_risk import calculate_collision_risk
from trasim_simplified.core.agent.fuzzy_logic import FuzzyLogic
from trasim_simplified.core.agent.mpc_solver import MPC_Solver
from trasim_simplified.core.agent.ref_path import ReferencePath
from trasim_simplified.core.agent.utils import get_xy_quintic, get_y_guess, get_x_guess
from trasim_simplified.core.constant import RouteType, VehSurr, TrajPoint, GameRes

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class Game_Vehicle(AgentBase, ABC):
    def __init__(self, lane: 'LaneAbstract', type_: int, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.T_b = 1.5
        """临界期望时距"""
        self.rho = 0.5
        """激进系数"""

        self.lc_acc_gain = []
        """换道加速度增益"""

        self.Rho_ori = 0.5

        self.e_route_threshold = 0.5

        self.rho_hat_s = {}  # 储存车辆id_对应的rho_hat

        self.lc_time_min = 1
        self.lc_time_max = 10

    def cal_safe_cost(self, traj, other_traj, k_sf=1):
        """计算与其他轨迹的安全成本"""
        if other_traj is None:
            return -np.inf
        assert len(traj) == len(other_traj), f"The length of traj is not equal to the other_traj."
        # 获取同一车道的轨迹，简化为如果横向y距离小于车身宽度，则认为在同一车道
        indexes = np.where(np.abs(traj[:, 3] - other_traj[:, 3]) < self.width)[0]
        traj = traj[indexes]
        other_traj = other_traj[indexes]

        # 最大cost求解
        if len(traj) == 0:
            return -np.inf

        # 计算安全成本
        safe_cost_list = []
        for point1, point2 in zip(traj, other_traj):
            gap = point2[0] - point1[0]
            v = point1[1]
            if -self.length < gap < self.length:
                safe_cost_list.append(np.inf)
                continue
            elif gap < 0:
                gap = point1[0] - point2[0]
                v = point2[1]
            safe_cost_list.append(k_sf * (self.T_desire * v - (gap - self.length)))

        safe_cost = max(safe_cost_list)
        return safe_cost

    def _get_rho_hat_s(self, vehicles):
        rho_hat_s = []
        for v in vehicles:
            if v is not None:
                rho_hat_s.append(self.rho_hat_s.get(v.ID, 0.5))
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
        # yaw = np.arctan2(np.diff(traj[:, 3]), np.diff(traj[:, 0]))
        # d_yaw_max = max(abs(np.diff(yaw) / self.dt))

        com_cost = k_c * (ddx_max + ddy_max)  # 舒适性计算
        # # 效率计算 单位时间内的平均速度-初始速度
        speed = np.sqrt(traj[:, 1] ** 2 + traj[:, 4] ** 2)
        eff_cost = k_e * np.mean(speed) - speed[0]  # 效率计算

        def cost(rho_):
            total_cost = ((1 - rho_) * safe_cost + rho_ * (k_com * com_cost + k_eff * eff_cost))
            return total_cost

        if return_lambda:
            return cost

        return cost(rho)

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

    @abc.abstractmethod
    def update_rho_hat(self):
        pass

    def pred_lc_risk(self, time_len=3):
        traj_cp, traj_lp, traj_lr, traj_rp, traj_rr = [
            self.pred_net.pred_traj(veh.pack_veh_surr(), type_="net", time_len=time_len)
            if veh is not None else None
            for veh in [self.leader, self.lf, self.lr, self.rf, self.rr]
        ]

        if self.left_lane is not None:
            if abs(traj_lp[0].x - self.x) < 7:
                traj_lp, traj_lr

            traj_ev_left, _ = self.pred_self_traj(
                target_lane=self.left_lane, PC_traj=traj_lp, time_len=time_len, to_ndarray=False
            )
            ev_lp_ttc_2d = calculate_collision_risk(traj_ev_left, traj_lp) if traj_lp is not None else np.inf
            ev_lr_ttc_2d = calculate_collision_risk(traj_ev_left, traj_lr) if traj_lr is not None else np.inf
            ev_cp_ttc_2d = calculate_collision_risk(traj_ev_left, traj_cp) if traj_cp is not None else np.inf
            min_ttc_2d_left = min(np.min(ev_lp_ttc_2d), np.min(ev_lr_ttc_2d), np.min(ev_cp_ttc_2d))
        else:
            min_ttc_2d_left = -np.inf
        if self.right_lane is not None:
            traj_ev_right, _ = self.pred_self_traj(
                target_lane=self.right_lane, PC_traj=traj_rp, time_len=time_len, to_ndarray=False
            )
            ev_rp_ttc_2d = calculate_collision_risk(traj_ev_right, traj_rp) if traj_rp is not None else np.inf
            ev_rr_ttc_2d = calculate_collision_risk(traj_ev_right, traj_rr) if traj_rr is not None else np.inf
            ev_cp_ttc_2d = calculate_collision_risk(traj_ev_right, traj_cp) if traj_cp is not None else np.inf
            min_ttc_2d_right = min(np.min(ev_rp_ttc_2d), np.min(ev_rr_ttc_2d), np.min(ev_cp_ttc_2d))
        else:
            min_ttc_2d_right = -np.inf
        return min_ttc_2d_left, min_ttc_2d_right

    def cal_route_cost(self):
        # 目标路径
        target_direction = 0
        if self.route_type == RouteType.diverge and self.lane.index not in self.destination_lane_indexes:
            current_lane_index = self.lane.index
            least_lane_index = min(self.destination_lane_indexes)
            n_need_lc = least_lane_index - current_lane_index
            d_r = self.lane.road.end_weaving_pos - self.x
            t_r = d_r / self.v
            e_r = max([0, 1 - d_r / (n_need_lc * 20), 1 - t_r / (n_need_lc * 3)])  # 预期20m一次换道，3s一次换道
            target_direction = 1
        elif self.route_type == RouteType.merge and self.lane.index not in self.destination_lane_indexes:
            current_lane_index = self.lane.index
            max_lane_index = max(self.destination_lane_indexes)
            n_need_lc = current_lane_index - max_lane_index
            d_r = self.lane.road.start_weaving_pos - self.x
            t_r = d_r / self.v
            e_r = max([0, 1 - d_r / (n_need_lc * 20), 1 - t_r / (n_need_lc * 3)])
            target_direction = -1
        else:
            e_r = 0
        return target_direction, e_r

    def pred_lc_benefit(self):
        target_lane, res = self.lc_model.step(self.pack_veh_surr())
        acc_gain = res["acc_gain"]  # left, right换道的加速度收益
        left_acc_gain = acc_gain[0]
        right_acc_gain = acc_gain[1]
        return left_acc_gain, right_acc_gain  # TODO：可能要归一化

    def lc_intention_judge(self):
        self.left_ttc, self.right_ttc = self.pred_lc_risk()
        if self.lane_changing:
            return
        route_direction, e_r = self.cal_route_cost()
        # 计算换道收益
        left_gain, right_gain = self.pred_lc_benefit()
        # 处于跟驰状态或有换道中有碰撞风险，重新计算目标车道
        if not self.lane_changing:
            if route_direction == 0:
                if left_gain >= right_gain:
                    self.lc_direction = -1
                    self.lc_acc_benefit = left_gain
                    self.lc_ttc_risk = self.left_ttc
                else:
                    self.lc_direction = 1
                    self.lc_acc_benefit = right_gain
                    self.lc_ttc_risk = self.right_ttc
            else:
                if route_direction == -1:
                    self.lc_direction = -1
                    self.lc_acc_benefit = left_gain
                    self.lc_ttc_risk = self.left_ttc
                else:
                    self.lc_direction = 1
                    self.lc_acc_benefit = right_gain
                    self.lc_ttc_risk = self.right_ttc
            self.lc_route_desire = e_r

            print("ID:", self.ID, "lc_direction:", self.lc_direction,
                  "benefit", self.lc_acc_benefit, "risk:", self.lc_ttc_risk)

    def _no_car_correction(self, gap):
        """判别TR、TF、PC是否存在，若不存在则设置为不影响换道博弈的虚拟车辆
        :gap: 车辆间距
        """
        if self.lc_direction == 0:
            raise ValueError("lc_direction is 0, please check the code.")
        if gap == -1:  # 后向搜索
            if self.lc_direction == 1:
                TF = self.rr
                TR = self.rr.r if self.rr is not None else None
            else:
                TF = self.lr
                TR = self.lr.f if self.lr is not None else None
        elif gap == 1:  # 前向搜索
            if self.lc_direction == 1:
                TF = self.rf.f if self.rf is not None else None
                TR = self.rf
            else:
                TF = self.lf.f if self.lf is not None else None
                TR = self.lf
        elif gap == 0:  # 相邻搜索
            if self.lc_direction == 1:
                TF = self.rf
                TR = self.rr
            else:
                TF = self.lr
                TR = self.lf
        else:
            raise ValueError("gap should be -1, 0 or 1, please check the code.")

        PC = self.f
        if TR is None:
            position = self.position + np.array([-1e10, - self.lc_direction * self.lane.width])
            TR = Game_O_Vehicle(self.lane, self.type, - self.ID, self.length)
            TR.x = position[0]
            TR.y = position[1]
        if TF is None:
            position = self.position + np.array([1e10, - self.lc_direction * self.lane.width])
            TF = Game_O_Vehicle(self.lane, self.type, - self.ID, self.length)
            TF.x = position[0]
            TF.y = position[1]
        if PC is None:
            position = self.position + np.array([1e10, 0])
            PC = Game_O_Vehicle(self.lane, self.type, - self.ID, self.length)
            PC.x = position[0]
            PC.y = position[1]
        return TR, TF, PC


class Game_A_Vehicle(Game_Vehicle):
    """自动驾驶车辆"""
    def __init__(self, lane: 'LaneAbstract', type_: int, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.TF_co = 1  # TF自动驾驶的公平协作程度 [0, 1]
        """TR, TF, PC"""

        self.LC_pred_traj_df: Optional[pd.DataFrame] = None
        """换道车对该车轨迹的预测"""

        self.N_MPC = 20
        self.mpc_solver: Optional[MPC_Solver] = None

        self.lc_step_length = -1  # 换道步数
        self.lc_end_step = -1  # 换道完成时的仿真步数
        self.lc_start_step = -1  # 换道开始时的仿真步数
        self.lc_entered_step = -1  # 车辆中心进入目标车道的仿真步数
        self.lc_step_conti = 0  # 如果正在换道，持续的换道步数

        self.lc_time_max = 10  # 换道最大时间
        self.lc_time_min = 1  # 换道最小时间

        self.opti_game_res: Optional[GameRes] = None
        self.lc_traj: Optional[np.ndarray] = None

    def lc_intention_judge(self):
        # 处于跟驰状态或有换道中有碰撞风险，重新计算目标车道
        super().lc_intention_judge()

    def lc_decision_making(self):
        """判断是否换道"""
        if self.no_lc:
            return
        # 根据上一时间步的TR行为策略更新rho_hat
        if self.is_gaming:
            self.update_rho_hat()
        self.state = self.get_state_for_traj()
        if self.lane_changing or self.lc_direction != 0:
            # LC轨迹优化，如果博弈选择不换道，需要重新更新self.lane_changing以及self.target_lane
            if self.lane_changing and self.lane != self.target_lane:
                # if self.lc_direction == -1:
                #     ttc = self.left_ttc
                # else:
                #     ttc = self.right_ttc
                # if ttc > self.ttc_star:
                #     return
                try:
                    self.stackel_berg()
                except ValueError:
                    pass
                return
            if not self.lane_changing and self.lc_direction != 0:
                self.stackel_berg()

    def cal_vehicle_control(self):
        """横向控制"""
        if self.lane == self.target_lane:
            # 恢复TR、TF的TIME_WANTED
            TR = self.opti_game_res.TR
            TR.clear_game_stra()
            TF = self.opti_game_res.TF
            TF.clear_game_stra()
            self.is_gaming = False

        if self.lane.step_ > self.lc_end_step:
            # 换道完成
            self.lane_changing = False

        print("ID:", self.ID, "lane_changing", self.lane_changing,
              "lc_end_step", self.lc_end_step, "current_step", self.lane.step_)
        if self.lane_changing is False:
            delta = self.cf_lateral_control()
            acc = self.cf_model.step(self.pack_veh_surr())
        else:
            # 横向控制
            acc, delta = self.mpc_solver.step_mpc()  # 默认计算出的符合约束的加速度和转向角
            self.lc_step_conti += 1

        self.next_acc = acc
        self.next_delta = delta

    def update_rho_hat(self):
        """求解rho的范围，入股TR的rho_hat不在范围内，更新TR的rho_hat"""
        if (self.opti_game_res is None or
                (self.lane_changing and self.lane.step_ >= self.lc_entered_step)):
            return
        TR = self.opti_game_res.TR
        TF = self.opti_game_res.TF
        if isinstance(TR, Game_H_Vehicle):
            # 第一次估计TR的rho_hat默认为0.5
            self.rho_hat_s[TR.ID] = self.rho_hat_s.get(TR.ID, 0.5)
            TR_real_stra = self.opti_game_res.TR_real_stra
            ego_stra = self.opti_game_res.EV_stra
            TR_traj_real, _ = TR.pred_self_traj(ego_stra, TR_real_stra)

            TR_traj, _ = TR.pred_self_traj(
                ego_stra, self.opti_game_res.EV_opti_series["TR_stra"], to_ndarray=True)
            TF_traj, _ = TF.pred_self_traj(
                ego_stra, self.opti_game_res.EV_opti_series["TF_stra"], to_ndarray=True)
            ego_traj = self.opti_game_res.EV_opti_traj[:round(ego_stra / self.dt) + 1, :]

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
                    print(lower_bound, upper_bound)
                    # 更新TR的rho_hat
                    if lower_bound <= self.rho_hat_s[TR.ID] <= upper_bound:
                        continue
                    else:
                        self.rho_hat_s[TR.ID] = (lower_bound + upper_bound) / 2

    def cal_ref_path(self, x_opt):
        times = np.arange(0, (self.lc_step_length + 0.5) * self.dt, self.dt)
        # x, vx, ax, y, vy, ay
        ref_path_ori = np.vstack([np.array(get_xy_quintic(x_opt, t)) for t in times])
        lc_end_state = ref_path_ori[-1, :]
        x, vx, ax, y, vy, ay = list(lc_end_state)
        times_after = np.arange(self.dt, (self.N_MPC + 0.5) * self.dt, self.dt).reshape(-1, 1)
        ref_path = np.vstack(
            [
                ref_path_ori,
                np.hstack([
                    x + vx * times_after + 0.5 * ax * times_after ** 2,
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

    def reset_lc_game(self):
        """重置博弈状态"""
        self.is_gaming = False
        self.lane_changing = False
        self.lc_direction = 0
        self.target_lane = self.lane
        self.lc_traj = None
        self.opti_game_res = None
        self.mpc_solver = None

    def stackel_berg(self):
        """基于stackelberg主从博弈理论的换道策略，计算得到参考轨迹（无论是否换道）"""
        # 计算不同策略（换道时间）的最优轨迹和对应效用值
        # cal_game_matrix需要更新周边车辆的策略，stackel_berg函数只更新自身的策略
        game_res_list = self.cal_game_matrix()
        # 获取最优策略
        game_cost_list = [res.EV_cost if res.EV_cost is not None else np.inf for res in game_res_list]
        min_cost_idx = np.argmin(game_cost_list)
        opti_game_res = game_res_list[min_cost_idx]
        if opti_game_res.EV_cost is None or np.isinf(opti_game_res.EV_cost):
            self.reset_lc_game()

        # 获取最优策略对应的轨迹
        opti_df = opti_game_res.EV_opti_series
        # a0~a5, b0~b5, step, y1, ego_stra, TF_stra, TR_stra, PC_stra, ego_cost, TF_cost, TR_cost, PC_cost
        # 选择效用值最小的策略
        x_opt = opti_df.iloc[: 12].to_numpy().astype(float)
        T_opt = opti_df.iloc[14]
        step_opt = int(opti_df.iloc[12])

        # 获取T_opt对应的不换道轨迹
        # traj, PC_traj = self.pred_self_traj(T_opt, stra=self.time_wanted)
        # cost = self.cal_cost_by_traj(traj, PC_traj=PC_traj)

        lc_result = self.opti_quintic_no_lc(
            T_opt, self.y, self.lane.y_center - self.lane.width / 2, self.lane.y_center + self.lane.width / 2,
            opti_game_res.PC
        )
        cost = lc_result[0]

        route_cost = self.cal_route_cost()

        if cost + route_cost < opti_game_res.EV_cost:
            self.reset_lc_game()

        self.opti_game_res = opti_game_res
        self.opti_game_res.TR.set_game_stra(self.opti_game_res.TR_real_stra)
        if isinstance(self.opti_game_res.TF, Game_A_Vehicle):
            self.opti_game_res.TF.set_game_stra(self.opti_game_res.TF_stra)

        self.lc_step_length = np.round(T_opt / self.dt).astype(int)
        self.lc_start_step = self.lane.step_
        self.lc_end_step = self.lc_start_step + self.lc_step_length
        self.lc_entered_step = self.lc_start_step + step_opt
        self.lc_traj = self.cal_ref_path(x_opt)

        self.lane_changing = True
        self.is_gaming = True
        self.target_lane = self.left_lane if self.lc_direction == -1 else self.right_lane
        self.game_time_wanted = self.time_wanted  # 不变

    def get_y_constraint(self, is_lc):
        lane_center_y = self.lane.y_center
        lane_width = self.lane.width

        if is_lc:
            y1 = lane_center_y + (- self.lc_direction) * lane_width
            y_limit_low = min(lane_center_y, y1) - lane_width / 2
            y_limit_up = max(lane_center_y, y1) + lane_width / 2
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
        game_res_list = []
        for gap in [-1, 0, 1]:
            TR, TF, PC = self._no_car_correction(gap=gap)
            TR_real_stra, TF_stra, EV_stra = None, None, None
            TR_real_cost, TF_cost, EV_cost = None, None, None
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
                    if len(cost_df_temp) == 0:
                        game_res_list.append(GameRes(None, TF, TR, PC, None, None,
                                                     None, None, None, None, None))
                        continue
                    min_TR_cost_idx = cost_df_temp.groupby('ego_stra')['TR_cost'].idxmin()  # 找到最小的TR_cost值
                    min_cost_idx = cost_df_temp.loc[min_TR_cost_idx]["ego_cost"].idxmin()  # 找到最小的cost值
                    EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                    EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
                    EV_opti_series = cost_df_temp.loc[min_cost_idx]

                    # 根据真实的TR成本函数计算真正的TR_stra（站在TR的立场上）
                    # cost_df_tr = cost_df[(cost_df["TR_2_EV_pred_cost"] != np.inf)]
                    # min_TR_cost_idx_ = cost_df_tr.groupby('ego_stra')['TR_real_cost'].idxmin()  # 找到最小的TR_cost值
                    # min_cost_idx_ = cost_df_tr.loc[min_TR_cost_idx_]["TR_2_EV_pred_cost"].idxmin()
                    # temp_df = cost_df[(cost_df["ego_stra"] == cost_df_tr.loc[min_cost_idx_]["ego_stra"])]
                    # min_TR_real_cost_idx = temp_df["TR_real_cost"].idxmin()

                    # 更新TR的策略
                    # TR_real_stra = temp_df.loc[min_TR_real_cost_idx]["TR_stra"]
                    # TR_real_cost = temp_df.loc[min_TR_real_cost_idx]["TR_real_cost"]

                    min_TR_cost_idx_real = cost_df_temp.groupby('ego_stra')['TR_real_cost'].idxmin()  # 找到最小的TR_cost值
                    min_cost_idx_real = cost_df_temp.loc[min_TR_cost_idx_real]["ego_cost"].idxmin()  # 找到最小的cost值
                    TR_real_stra = cost_df_temp.loc[min_cost_idx_real]["TR_stra"]
                    TR_real_cost = cost_df_temp.loc[min_cost_idx_real]["TR_real_cost"]
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
                    if len(cost_df_temp) == 0:
                        game_res_list.append(GameRes(None, TF, TR, PC, None, None, None,
                                                     None, None, None, None))
                        continue
                    # 找到最小的TR_cost值
                    min_TR_cost_idx = cost_df_temp.groupby(by=["ego_stra", "TF_stra"])['total_cost'].idxmin()
                    min_cost_idx = cost_df_temp.loc[min_TR_cost_idx]["total_cost"].idxmin()  # 找到最小的cost值

                    TF_stra = cost_df_temp.loc[min_cost_idx]["TF_stra"]
                    EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                    TF_cost = cost_df_temp.loc[min_cost_idx]["TF_cost"]
                    EV_cost = cost_df_temp.loc[min_cost_idx]["cost"]
                    EV_opti_series = cost_df_temp.loc[min_cost_idx]

                    # 根据真实的TR成本函数计算真正的TR_stra（站在TR的立场上）
                    TF_co_pred = TR.get_co_hat_s([TF])[0]
                    cost_df["total_cost_TR_pred"] = (
                            cost_df["ego_cost"] + TF_co_pred * cost_df["TF_cost"])
                    cost_df_temp = cost_df[cost_df["total_cost_TR_pred"] != np.inf]
                    # 找到最小的TR_cost值
                    min_TR_cost_idx_ = cost_df_temp.groupby(by=["ego_stra", "TF_stra"])['total_cost_TR_pred'].idxmin()
                    min_cost_idx_ = cost_df_temp.loc[min_TR_cost_idx_]["total_cost_TR_pred"].idxmin()

                    temp_df = cost_df[(cost_df["ego_stra"] == cost_df_temp.loc[min_cost_idx_]["ego_stra"]) &
                                      (cost_df["TF_stra"] == cost_df_temp.loc[min_cost_idx_]["TF_stra"]) &
                                      (cost_df["total_cost_TR_pred"] != np.inf)]
                    min_TR_real_cost_idx = temp_df["TR_real_cost"].idxmin()

                    # 更新TR的策略
                    TR_real_stra = temp_df.loc[min_TR_real_cost_idx]["TR_stra"]
                    TR_real_cost = temp_df.loc[min_TR_real_cost_idx]["TR_real_cost"]
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
                    if len(cost_df_temp) == 0:
                        game_res_list.append(GameRes(None, TF, TR, PC, None, None, None, None,
                                                     None, None, None))
                        continue
                    min_cost_idx = cost_df_temp["total_cost"].idxmin()  # 找到最小的cost值

                    # 更新TR的策略
                    TR_real_stra = cost_df_temp.loc[min_cost_idx]["TR_stra"]
                    EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                    TR_real_cost = cost_df_temp.loc[min_cost_idx]["TR_cost"]
                    EV_cost = cost_df_temp.loc[min_cost_idx]["cost"]
                    EV_opti_series = cost_df_temp.loc[min_cost_idx]
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

                    cost_df["total_cost"] = (
                            cost_df["cost"] + TR.TF_co * cost_df["TR_cost"] + TF.TF_co * cost_df["TF_cost"]
                    )
                    cost_df_temp = cost_df[cost_df["total_cost"] != np.inf]
                    if len(cost_df_temp) == 0:
                        game_res_list.append(GameRes(None, TF, TR, PC, None, None, None,
                                                     None, None, None, None))
                        continue
                    min_cost_idx = cost_df_temp["total_cost"].idxmin()  # 找到最小的cost值

                    TR_real_stra = cost_df_temp.loc[min_cost_idx]["TR_stra"]
                    TF_stra = cost_df_temp.loc[min_cost_idx]["TF_stra"]
                    EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                    TR_real_cost = cost_df_temp.loc[min_cost_idx]["TR_cost"]
                    TF_cost = cost_df_temp.loc[min_cost_idx]["TF_cost"]
                    EV_cost = cost_df_temp.loc[min_cost_idx]["cost"]
                    EV_opti_series = cost_df_temp.loc[min_cost_idx]
                else:
                    print(f"车辆类型错误：TR: {type(TR)}, TF: {type(TF)}")
                    raise NotImplementedError("未知的车辆类型")
            except cvxpy.error.SolverError:
                return GameRes(None, TF, TR, PC, None, None, None,
                               None, None, None, None)

            game_res_list.append(GameRes(
                cost_df, TF, TR, PC, EV_stra, TF_stra, TR_real_stra,
                EV_cost, TF_cost, TR_real_cost, EV_opti_series
            ))

        return game_res_list

    def _cal_cost_df(self, stra_product, cal_TF_cost, cal_TR_cost, cal_PC_cost, TF, TR, PC):
        stra_product = list(stra_product)
        num = len(stra_product)
        # self.LC_pred_traj_df = pd.DataFrame(
        #     data=np.zeros((num, 10), dtype=object),
        #     columns=[*[f"{name}_{type_}" for type_ in ["stra", "traj"]
        #                for name in ["ego", "TF", "TR", "PC"]]] + ["TR_2_EV_pred_cost", "TR_2_TF_pred_cost"]
        # )
        data = np.vstack(
            list(
                list(self.cal_cost(
                    ego_stra, TF=TF, TR=TR, PC=PC,
                    TF_stra=TF_stra, TR_stra=TR_stra, PC_stra=PC_stra,
                    cal_TF_cost=cal_TF_cost, cal_TR_cost=cal_TR_cost, cal_PC_cost=cal_PC_cost, index=i
                )) for i, [ego_stra, TF_stra, TR_stra, PC_stra] in enumerate(stra_product)
            )
        )
        cost_df = pd.DataFrame(
            data,
            columns=[*[f"{i}{j}" for i in ["a", "b"] for j in range(6)], *["step", "y1"],
                     *[f"{name}_{type_}" for type_ in ["stra", "cost", "real_cost"]
                       for name in ["ego", "TF", "TR", "PC"]]] + ["TR_2_EV_pred_cost", "TR_2_TF_pred_cost"]
        )
        return cost_df

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
        # if lc_result[0] == np.inf:
        #     # [*[f"{i}{j}" for i in ["a", "b"] for j in range(6)], *["step", "y1"],
        #     #  *[f"{name}_{type_}" for type_ in ["stra", "cost", "real_cost"]
        #     #    for name in ["ego", "TF", "TR", "PC"]]] + ["TR_2_EV_pred_cost", "TR_2_TF_pred_cost"]
        #     return np.hstack([lc_result[1:], [TF_stra, TR_stra, PC_stra],
        #                       [lc_result[0], np.nan, np.nan, np.nan,
        #                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]).reshape(1, -1)

        time_steps = np.arange(0, stra + self.dt / 2, self.dt)
        if lc_result[0] == np.inf:
            cost, traj = lc_result[0], np.vstack(
                [np.array(get_xy_quintic(lc_result[1:13], t)) for t in time_steps]
            )
        else:
            cost = lc_result[0]
            traj, _ = self.pred_self_traj(stra)

        # self.LC_pred_traj_df.loc[index, 'ego_stra'] = stra
        # self.LC_pred_traj_df.loc[index, 'TF_stra'] = TF_stra
        # self.LC_pred_traj_df.loc[index, 'TR_stra'] = TR_stra
        # self.LC_pred_traj_df.loc[index, 'PC_stra'] = PC_stra
        #
        # self.LC_pred_traj_df.loc[index, 'ego_traj'] = traj
        # self.LC_pred_traj_df.loc[index, 'TF_traj'] = TF_traj
        # self.LC_pred_traj_df.loc[index, 'TR_traj'] = TR_traj
        # self.LC_pred_traj_df.loc[index, 'PC_traj'] = PC_traj

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
        TR_2_EV_pred_cost = np.inf  # TR对EV的效用估计
        TR_2_TF_pred_cost = np.inf  # TR对TF的效用估计
        # if TR is not None and isinstance(TR, Game_H_Vehicle):
        #     EV_rho_hat = TR._get_rho_hat_s([self])[0]
        #     TR_2_EV_pred_cost = self.cal_cost_by_traj(
        #         traj, PC_traj=PC_traj, TR_traj=TR_traj, TF_traj=TF_traj, rho=EV_rho_hat
        #     )
        #     if TF is not None and isinstance(TF, Game_A_Vehicle):
        #         TF_rho_hat = TR._get_rho_hat_s([TF])[0]
        #         TR_2_TF_pred_cost = TR.cal_cost_by_traj(
        #             TR_traj, PC_traj=TR_PC_traj, rho=TF_rho_hat
        #         )

        # [*[f"{i}{j}" for i in ["a", "b"] for j in range(6)], *["step", "y1"],
        #  *[f"{name}_{type_}" for type_ in ["stra", "cost", "real_cost"]
        #    for name in ["ego", "TF", "TR", "PC"]]] + ["TR_2_EV_pred_cost", "TR_2_TF_pred_cost"]

        # lc_result: np.hstack([[cost_value], x_opt[:12], [step, y1, T]])
        result = np.hstack(
            [lc_result[1:], [TF_stra, TR_stra, PC_stra],
             [cost, TF_cost, TR_cost, PC_cost,
              cost, TF_real_cost, TR_real_cost, PC_real_cost,
              TR_2_EV_pred_cost, TR_2_TF_pred_cost]]
        ).reshape(1, -1)
        return result

    def opti_quintic_no_lc(self, T, y1, y_limit_low, y_limit_up, PC: Game_Vehicle = None):
        """[cost_value], x_opt, [T, step, y1]"""
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
        b_ineq4 = np.array([self.max_speed] * len(A_ineq3))
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
        vehicles = [PC]
        [(PC_traj, l_PC, PC_PC_traj)] = \
            self.pred_traj_s(
                T,
                vehicles,
                stra_s=[None]
            )
        xt_PC, dxt_PC = PC_traj[:, 0], PC_traj[:, 1]

        # 提取换道至目标车道初始时刻
        y, yf, dyf, ddyf = get_y_guess(T, self.state, y1)
        A_y = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])

        A_ineq_7 = np.array([
            [1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])  # 过线点前位置
        x_pos = A_ineq_7 @ x[: 6]  # 过线点前位置

        # PC
        dxt = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps])
        v = dxt @ x[:6]
        PC_rear_x = xt_PC - l_PC
        b_ineq_7_part1 = PC_rear_x - v * self.time_safe
        constraints += [x_pos <= b_ineq_7_part1]
        # b_ineq_7_part2 = PC_rear_x - v * self.T_desire - v * (v - dxt_PC[:step]) / acc_temp
        b_ineq_7_part2 = PC_rear_x - v * self.time_wanted
        constraints += [x[12] >= x_pos - b_ineq_7_part2]

        constraints += [x[13] == 0]
        constraints += [x[14] == 0]

        # 安全性
        safe_cost = cvxpy.sum(x[12: 15])

        # 舒适性：横纵向jerk目标
        k_c = 1
        A_xy_acc = np.array([
            [0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3] for t in time_steps
        ])
        com_cost = k_c * cvxpy.max(cvxpy.abs(A_xy_acc @ x[:6]))
        com_cost += k_c * cvxpy.max(cvxpy.abs(A_xy_acc @ x[6:12]))

        # 效率
        k_e = 1
        A_dxy = np.array([
            [0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps
        ])
        vx = A_dxy @ x[:6]
        eff_cost = k_e * cvxpy.max(cvxpy.abs(cvxpy.mean(vx) - vx[0]))
        rho_ = self.rho
        cost = (1 - rho_) * safe_cost + rho_ * (0.5 * com_cost + 0.5 * eff_cost)

        result = np.hstack([[np.inf], np.tile(np.nan, 12), [np.nan, y1, T]])
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        try:
            cost_value = prob.solve(verbose=False, solver=cvxpy.GUROBI, reoptimize=True)
            if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
                x_opt = x.value
                result = np.hstack([[cost_value], x_opt[:12], [None, y1, T]])
        except cvxpy.error.SolverError:
            pass
        return result


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
        b_ineq4 = np.array([self.max_speed] * len(A_ineq3))
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
        vehicles = [TR, TF, PC]
        (TF_traj, l_TF, TF_PC_traj), (TR_traj, l_TR, TR_PC_traj), (PC_traj, l_PC, PC_PC_traj) =\
            self.pred_traj_s(
                T,
                vehicles,
                stra_s=[TF_stra, TR_stra, PC_stra]
            )
        xt_TF, dxt_TF = TF_traj[:, 0], TF_traj[:, 1]
        xt_TR, dxt_TR = TR_traj[:, 0], TR_traj[:, 1]
        xt_PC, dxt_PC = PC_traj[:, 0], PC_traj[:, 1]

        # 提取换道至目标车道初始时刻
        y, yf, dyf, ddyf = get_y_guess(T, self.state, y1)
        A_y = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])
        Y = A_y @ y
        # 进入目标车道的初始时刻 # ATTENTION：由于为车头的中点的y坐标，因此初始时刻为近似
        if Y[0] < Y[-1]:
            step = (Y[Y <= (y_limit_low + y_limit_up) / 2]).shape[0]  # 进入目标车道的初始时刻
        else:
            step = (Y[Y >= (y_limit_low + y_limit_up) / 2]).shape[0]  # 进入目标车道的初始时刻
        lc_before_times = time_steps[:step]
        lc_after_times = time_steps[step:]

        # acc_temp = 2 * np.sqrt(self.acc_desire * self.dec_desire)

        A_ineq_7 = np.array([
            [1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in lc_before_times])  # 过线点前位置
        x_pos_before = A_ineq_7 @ x[: 6]  # 过线点前位置
        A_ineq_8 = np.array([
            [1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in lc_after_times])  # 过线点后位置
        x_pos_after = A_ineq_8 @ x[: 6]  # 过线点后位置

        # PC
        dxt_before = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in lc_before_times])
        v = dxt_before @ x[:6]
        PC_rear_x = xt_PC[:step] - l_PC
        b_ineq_7_part1 = PC_rear_x - v * self.time_safe
        constraints += [x_pos_before <= b_ineq_7_part1]
        # b_ineq_7_part2 = PC_rear_x - v * self.T_desire - v * (v - dxt_PC[:step]) / acc_temp
        b_ineq_7_part2 = PC_rear_x - v * self.time_wanted
        constraints += [x[12] >= x_pos_before - b_ineq_7_part2]

        # TF
        dxt_after = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in lc_after_times])
        v = dxt_after @ x[:6]
        TF_rear_x = xt_TF[step:] - l_TF
        b_ineq_8_part1 = TF_rear_x - v * self.time_safe
        constraints += [x_pos_after <= b_ineq_8_part1]
        # b_ineq_8_part2 = TF_rear_x - v * self.T_desire - v * (v - dxt_TF[step:]) / acc_temp
        b_ineq_8_part2 = TF_rear_x - v * self.time_wanted
        constraints += [x[13] >= x_pos_after - b_ineq_8_part2]

        # TR
        TR_head_x = xt_TR[step:] + l_TR
        TR_v = dxt_TR[step:]
        b_ineq_9_part1 = TR_head_x + self.time_safe * TR_v
        constraints += [x_pos_after >= b_ineq_9_part1]
        # b_ineq_9_part2 = TR_head_x + self.T_desire * TR_v + TR_v * (TR_v - v) / acc_temp
        b_ineq_9_part2 = TR_head_x + self.time_wanted * TR_v
        constraints += [x[14] >= b_ineq_9_part2 - x_pos_after]

        # 安全性
        k_TP = k_TF = k_PC = 1
        safe_cost = cvxpy.sum(x[12: 15])

        # 舒适性：横纵向jerk目标
        k_c = 1
        # A_x_jerk = np.array([
        #     [0, 0, 0, 6, 24 * t, 60 * t ** 2] for t in time_steps
        # ])
        A_xy_acc = np.array([
            [0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3] for t in time_steps
        ])
        # assert np.all(np.linalg.eigvals(A_x_jerk.T @ A_x_jerk) >= 0), "Matrix P is not positive semi-definite."
        com_cost = k_c * cvxpy.max(cvxpy.abs(A_xy_acc @ x[:6]))
        com_cost += k_c * cvxpy.max(cvxpy.abs(A_xy_acc @ x[6:12]))
        # cost += k_c * cvxpy.quad_form(x[:6], A_x_jerk.T @ A_x_jerk, assume_PSD=True)

        # 效率
        k_e = 1
        A_dxy = np.array([
            [0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps
        ])
        vx = A_dxy @ x[:6]
        eff_cost = k_e * cvxpy.max(cvxpy.abs(cvxpy.mean(vx) - vx[0]))
        rho_ = self.rho
        cost = (1 - rho_) * safe_cost + rho_ * (0.5 * com_cost + 0.5 * eff_cost)
        # cost += k_e * ((A_dx @ x[:12] - self.target_speed).T @ (A_dx @ x[:12] - self.target_speed)) * self.dt
        # 如果上式不行，可以用下面的
        # target_speed_array = np.tile(self.target_speed, (len(time_steps), 1))
        # cost += k_e * (cvxpy.quad_form(x[:12], A_dx.T @ A_dx, assume_PSD=True)
        #                - x[:12].T @ A_dx.T @ target_speed_array
        #                - target_speed_array.T @ A_dx @ x[:12])

        result = np.hstack([[np.inf], np.tile(np.nan, 12), [np.nan, y1, T]])
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        try:
            cost_value = prob.solve(verbose=False, solver=cvxpy.GUROBI, reoptimize=True)
            if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
                x_opt = x.value
                result = np.hstack([[cost_value], x_opt[:12], [step, y1, T]])
        except cvxpy.error.SolverError:
            pass

        if return_other_traj:
            other_traj = [TF_traj, TR_traj, PC_traj, TF_PC_traj, TR_PC_traj, PC_PC_traj]
            return result, other_traj
        return result

    def pred_traj_s(self, T, vehicles: list[Game_Vehicle], stra_s) -> list[tuple[np.ndarray, float, np.ndarray]]:
        pred_traj_s = []
        for v, stra in zip(vehicles, stra_s):
            v = v if v is not None else self.f if self.lc_direction == 1 else self.r
            l = v.length
            traj, PC_traj = v.pred_self_traj(T, stra, to_ndarray=True)
            traj: np.ndarray = traj
            PC_traj: np.ndarray = PC_traj
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
        if self.no_lc:
            return
        if self.lc_ttc_risk <= self.ttc_star:
            return

        if self.lc_route_desire > 0.9 or self.lc_route_desire > np.random.uniform():
            decision = True
        elif self.lc_acc_benefit >= 0.9:
            decision = True
        else:
            lc_prob = self.fuzzy_logic.compute(self.lc_acc_benefit, self.lc_ttc_risk, self.rho)
            decision = 0.1 < lc_prob and lc_prob > np.random.uniform()

        if decision:
            self.lane_changing = True
            if self.lc_direction == 1 and self.right_lane is not None:
                self.target_lane = self.right_lane
            elif self.lc_direction == -1 and self.left_lane is not None:
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

    def update_rho_hat(self, vehicles: list[Game_Vehicle]):
        pass

    def get_strategies(self, is_lc: bool = False):
        if is_lc:
            # 换道方向
            strategy = [-1, 1]
        else:
            # 期望时距
            strategy = list(np.arange(0.1, 5.1, 0.5))
        return strategy


class Game_O_Vehicle(Game_H_Vehicle):
    def cal_cost_by_traj(self, traj, PC_traj=None, LC_traj=None, TR_traj=None, TF_traj=None,
                         rho=None, return_lambda=False, is_lc=False):
        def cost():
            return 0

        if return_lambda:
            return cost
        return cost()

    def get_strategies(self, is_lc: bool = False):
        return [self.T_safe]

    def lane_change_judge(self):
        pass

    def pred_self_traj(self, time_len, stra=None,
                       target_lane: "LaneAbstract" = None,
                       PC_traj=None, to_ndarray=True):
        step_num = round(time_len / self.dt) + 1
        if to_ndarray:
            traj_list = [self.get_traj_point().to_ndarray()] * step_num
        else:
            traj_list = [self.get_traj_point()] * step_num
        return np.array(traj_list), None
