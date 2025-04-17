# -*- coding: utf-8 -*-
# @time : 2025/3/22 22:13
# @Author : yzbyx
# @File : game_agent.py
# Software: PyCharm
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
from trasim_simplified.core.agent.utils import get_xy_quintic, get_y_guess
from trasim_simplified.core.constant import RouteType, VehSurr, TrajPoint, GameRes, TrajData, GapJudge, V_TYPE
from trasim_simplified.core.kinematics.lcm import LCModel_Mobil

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


def cal_other_cost(veh, cal_cost, rho_hat, traj, other_traj_s):
    if cal_cost:
        cost_lambda = veh.cal_cost_by_traj(traj, other_traj_s, return_lambda=True)
        cost_hat = cost_lambda(rho_hat) if cal_cost else np.nan
        real_cost = cost_lambda(veh.rho)
    else:
        cost_hat = np.nan
        real_cost = np.nan
        cost_lambda = None
    return cost_hat, real_cost, cost_lambda


class Game_Vehicle(AgentBase, ABC):
    fuzzy_logic = FuzzyLogic()

    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.rho = 0.5
        """激进系数"""

        self.rho_hat_s = {}  # 储存车辆id_对应的rho_hat

        self.opti_game_res: Optional[GameRes] = None
        self.lc_traj: Optional[np.ndarray] = None
        self.is_game_leader = False
        self.game_leader = None

        self.opti_gap: Optional[GapJudge] = None
        self.gap_res_list: Optional[list[GapJudge]] = None

        self.LC_CROSS_STEP = None
        self.LC_REAL_END_STEP = None

        self.single_stra = False

    def get_lane_indexes(self, y_s):
        """获取轨迹所在车道的索引"""
        lane_indexes = []
        for y in y_s:
            for lane in self.lane.road.lane_list:
                if lane.y_right < y <= lane.y_left:
                    lane_indexes.append(lane.index)
                    break
        return np.array(lane_indexes)

    def cal_safe_cost(self, traj, other_traj, k_sf=1):
        """计算与其他轨迹的安全成本"""
        if other_traj is None:
            return -np.inf
        assert len(traj) == len(other_traj), f"The length of traj is not equal to the other_traj."
        # 获取同一车道的轨迹，简化为如果横向y距离小于车身宽度，则认为在同一车道
        lane_indexes = self.get_lane_indexes(traj[:, 3])
        other_lane_indexes = self.get_lane_indexes(other_traj[:, 3])
        indexes = np.where(lane_indexes == other_lane_indexes)[0]
        traj = traj[indexes]
        other_traj = other_traj[indexes]

        # 最大cost求解
        if len(traj) == 0:
            return -np.inf

        # 计算安全成本
        safe_cost_list = []
        for point1, point2 in zip(traj, other_traj):
            dhw = point2[0] - point1[0]
            v = point1[1]
            if -self.length < dhw < self.length:
                safe_cost_list.append(np.inf)
                continue
            elif dhw < 0:
                dhw = point1[0] - point2[0]
                v = point2[1]
            safe_cost_list.append(k_sf * (self.time_wanted * v - (dhw - self.length)))

        safe_cost = max(safe_cost_list)
        return safe_cost

    def _get_rho_hat_s(self, vehicles):
        rho_hat_s = []
        for v in vehicles:
            if v is not None:
                if v.ID not in self.rho_hat_s:
                    self.rho_hat_s[v.ID] = 0.5
                rho_hat_s.append(self.rho_hat_s[v.ID])
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

    def cal_cost_by_traj(self, traj, other_traj_s,
                         rho=None, return_lambda=False,
                         route_cost=0):
        """
        :param traj: 预测轨迹 x, dx, ddx, y, dy, ddy
        :param other_traj_s: 其他车辆的轨迹
        :param rho: 激进系数
        :param route_cost: 路径成本
        :param return_lambda: 是否返回lambda函数
        """
        k_c = 0.5

        # 计算安全成本
        safe_cost_list = []
        for other_traj in other_traj_s:
            if other_traj is None:
                continue
            safe_cost_list.append(self.cal_safe_cost(traj, other_traj))

        safe_cost = max(safe_cost_list)
        # 舒适性计算 横纵向最大加速度、横摆角速度
        ddx_max = max(abs(traj[:, 2]))
        ddy_max = max(abs(traj[:, 5]))

        com_cost = k_c * (ddx_max + ddy_max)  # 舒适性计算
        # # 效率计算 单位时间内的平均速度-初始速度
        vx = traj[:, 1]
        eff_cost = vx[0] - np.mean(vx)  # 效率计算

        def cost(rho_):
            total_cost = (
                    rho_ * (safe_cost + com_cost) +
                    (1 - rho_) * eff_cost + route_cost
            )
            return total_cost

        if return_lambda:
            return cost

        return cost(rho)

    def get_strategies(self, is_lc: bool = False):
        if is_lc:
            # 换道时间
            strategy = itertools.product(np.arange(1, 10, 2), (1, 0))
        else:
            if self.single_stra:
                strategy = [self.time_wanted]
            else:
                gap = (self.time_wanted - 0.1) / 2
                # 期望时距
                strategy = list(np.arange(0.1, 0.1 + gap * 7, 1))
        return strategy

    def pred_lc_risk(self, time_len=3, lc_direction=None):
        """车辆当前时刻换道的风险/不换道的风险TTC"""
        # if lc_direction == -1:
        if self.left_lane is None:
            return -np.inf
        traj_cp, traj_lp, traj_lr = [
            self.pred_net.pred_traj(veh.pack_veh_surr(), type_="net", time_len=time_len)
            if veh is not None else None
            for veh in [self.leader, self.lf, self.lr]
        ]
        if ((traj_lp is not None and abs(traj_lp[0].x - self.x) < self.length) or
                traj_lr is not None and abs(traj_lr[0].x - self.x) < self.length):
            min_ttc_2d_left = -np.inf
        else:
            traj_ev_left, _ = self.pred_self_traj(
                target_lane=self.left_lane, PC_traj=traj_lp, time_len=time_len, to_ndarray=False
            )
            ev_lp_ttc_2d = calculate_collision_risk(traj_ev_left, traj_lp) if traj_lp is not None else np.inf
            ev_lr_ttc_2d = calculate_collision_risk(traj_ev_left, traj_lr) if traj_lr is not None else np.inf
            ev_cp_ttc_2d = calculate_collision_risk(traj_ev_left, traj_cp) if traj_cp is not None else np.inf
            min_ttc_2d_left = min(np.min(ev_lp_ttc_2d), np.min(ev_lr_ttc_2d), np.min(ev_cp_ttc_2d))

            # return min_ttc_2d_left

        # if lc_direction == 1:
        if self.right_lane is None:
            return -np.inf
        traj_cp, traj_rp, traj_rr = [
            self.pred_net.pred_traj(veh.pack_veh_surr(), type_="net", time_len=time_len)
            if veh is not None else None
            for veh in [self.leader, self.rf, self.rr]
        ]
        if ((traj_rp is not None and abs(traj_rp[0].x - self.x) < self.length) or
                (traj_rr is not None and abs(traj_rr[0].x - self.x) < self.length)):
            min_ttc_2d_right = -np.inf
        else:
            traj_ev_right, _ = self.pred_self_traj(
                target_lane=self.right_lane, PC_traj=traj_rp, time_len=time_len, to_ndarray=False
            )
            ev_rp_ttc_2d = calculate_collision_risk(traj_ev_right, traj_rp) if traj_rp is not None else np.inf
            ev_rr_ttc_2d = calculate_collision_risk(traj_ev_right, traj_rr) if traj_rr is not None else np.inf
            ev_cp_ttc_2d = calculate_collision_risk(traj_ev_right, traj_cp) if traj_cp is not None else np.inf
            min_ttc_2d_right = min(np.min(ev_rp_ttc_2d), np.min(ev_rr_ttc_2d), np.min(ev_cp_ttc_2d))

            # return min_ttc_2d_right

        # if lc_direction == 0:
        traj_cp = self.pred_net.pred_traj(self.leader.pack_veh_surr(), type_="net", time_len=time_len) \
            if self.leader is not None else None
        traj_ev_stay, _ = self.pred_self_traj(
            target_lane=self.lane, PC_traj=traj_cp, time_len=time_len, to_ndarray=False
        )
        ev_cp_ttc_2d = calculate_collision_risk(traj_ev_stay, traj_cp) if traj_cp is not None else np.inf
        min_ev_cp_ttc_2d = min(ev_cp_ttc_2d)
            # return min(ev_cp_ttc_2d)
        return min(min_ttc_2d_left, min_ttc_2d_right, min_ev_cp_ttc_2d)

        # raise ValueError("lc_direction should be -1, 0 or 1, please check the code.")

    def cal_route_cost(self, lane_index):
        # 目标路径
        target_direction = 0
        weaving_length = self.lane.road.end_weaving_pos - self.lane.road.start_weaving_pos
        d_r = self.lane.road.end_weaving_pos - self.x - weaving_length / 2
        t_r = d_r / self.v
        if self.route_type == RouteType.diverge and lane_index not in self.destination_lane_indexes:
            current_lane_index = lane_index
            least_lane_index = min(self.destination_lane_indexes)
            n_need_lc = least_lane_index - current_lane_index
            e_r = max([0, 1 - d_r / (n_need_lc * weaving_length), 1 - t_r / (n_need_lc * 10)])
            target_direction = 1
        elif self.route_type == RouteType.merge and lane_index not in self.destination_lane_indexes:
            current_lane_index = lane_index
            max_lane_index = max(self.destination_lane_indexes)
            n_need_lc = current_lane_index - max_lane_index
            e_r = max([0, 1 - d_r / (n_need_lc * weaving_length), 1 - t_r / (n_need_lc * 10)])
            target_direction = -1
        else:
            e_r = 0
        return target_direction, min(e_r, 1)

    def lc_intention_judge(self):
        if self.no_lc:
            return
        self.risk_2d = self.pred_lc_risk(lc_direction=self.lc_direction)
        if self.lane_changing and self.risk_2d >= self.ttc_star:
            return

        if self.lane_changing and self.risk_2d < self.ttc_star:
            self.lane_changing = False
            self.lc_direction = 0
            self.opti_gap = None

        if (self.opti_gap is not None and abs(self.y_c - self.target_lane.y_center) < 0.1
                and self.v_lat < 0.1 and abs(self.yaw) < 5 / 180 * np.pi):
            self.lane_changing = False
            self.lc_direction = 0
            self.opti_gap = None
            self.LC_REAL_END_STEP = self.lane.step_

        judge_res = []
        for adapt_time in np.arange(0, 3.1, 0.5):
            for lc_direction in [-1, 1]:
                for gap in [-1, 0, 1]:
                    if self.right_lane is None and lc_direction == 1:
                        continue
                    if self.left_lane is None and lc_direction == -1:
                        continue

                    target_lane_index = self.lane.index + lc_direction
                    if self.route_type == RouteType.mainline and target_lane_index not in self.destination_lane_indexes:
                        continue
                    if self.route_type == RouteType.merge and lc_direction != -1:
                        continue
                    if self.route_type == RouteType.diverge and lc_direction != 1:
                        continue

                    TR, TF, PC, is_exist = self._no_car_correction(gap, lc_direction)
                    if is_exist is False:
                        continue
                    TR_traj = self.pred_net.pred_traj(TR.pack_veh_surr(), time_len=adapt_time)
                    TF_traj = self.pred_net.pred_traj(TF.pack_veh_surr(), time_len=adapt_time)
                    PC_traj = self.pred_net.pred_traj(PC.pack_veh_surr(), time_len=adapt_time)
                    TR_end = TR_traj[-1]
                    TF_end = TF_traj[-1]
                    PC_end = PC_traj[-1]
                    # 判断3s（速度调整时间）内能否以舒适的加减速度达到可行换道位置，用于MOBIL模型判断
                    x_safe_min = TR_end.x + TR.length + self.time_safe * TR_end.vx
                    x_safe_max = TF_end.x - TF.length - self.time_safe * TF_end.vx
                    if not x_safe_min <= self.x <= x_safe_max:
                        acc_1 = 2 * (x_safe_max - self.x - self.v * adapt_time) / adapt_time ** 2
                        acc_2 = 2 * (x_safe_min - self.x - self.v * adapt_time) / adapt_time ** 2
                        # 若(acc_1, acc_2)与[self.acc_max, - self.dec_max]的交集不为空，
                        # 则选择加速度绝对值最小的加速度作为调整加速度
                        acc = np.array([acc_2, acc_1])
                        acc = np.clip(acc, - self.cf_model.get_com_dec(), self.cf_model.get_com_acc())
                        if acc[0] == acc[1]:
                            continue
                        if acc[0] <= 0 <= acc[1]:
                            target_acc = 0
                        elif acc[0] > 0:
                            target_acc = acc[0]
                        else:
                            target_acc = acc[1]
                    else:
                        target_acc = 0

                    # 与前车的安全性判断
                    v_f = self.v + target_acc * adapt_time
                    x_f = self.x + self.v * adapt_time + 0.5 * target_acc * adapt_time ** 2
                    if x_f > PC_end.x - PC.length - self.cf_model.get_safe_s0():
                        continue

                    ego_f = self.clone()
                    ego_f.x = x_f
                    ego_f.speed = v_f
                    PC_f = PC.clone()
                    PC_f.x = PC_end.x
                    PC_f.speed = PC_end.speed
                    TF_f = TF.clone()
                    TF_f.x = TF_end.x
                    TF_f.speed = TF_end.speed
                    TR_f = TR.clone()
                    TR_f.x = TR_end.x
                    TR_f.speed = TR_end.speed

                    if lc_direction == 1:
                        veh_surr = VehSurr(ego_f, cp=PC_f, rp=TF_f, rr=TR_f)
                    else:
                        veh_surr = VehSurr(ego_f, cp=PC_f, lp=TF_f, lr=TR_f)
                    try:
                        is_ok, acc_gain = LCModel_Mobil.mobil(lc_direction, veh_surr, return_acc_gain=True)
                    except:
                        is_ok, acc_gain = False, -np.inf

                    ego_traj = [TrajPoint(
                        x=self.x + self.v * t + 0.5 * target_acc * (t ** 2),
                        y=self.y + self.v_lat * t, speed=self.v + target_acc * t,
                        yaw=self.yaw, length=self.length, acc=target_acc,
                        width=self.width
                    ) for t in np.arange(0, adapt_time + self.dt / 2, self.dt)]
                    risk_pc = float(np.array(calculate_collision_risk(ego_traj, PC_traj))[-1])

                    _, e_r_stay = self.cal_route_cost(self.lane.index)
                    _, e_r_lc = self.cal_route_cost(self.lane.index + lc_direction)

                    judge_res.append(GapJudge(
                        lc_direction, target_acc, adapt_time,
                        self, TF=TF, TR=TR, PC=PC,
                        acc_gain=acc_gain,
                        ttc_risk=risk_pc,
                        route_gain=e_r_stay - e_r_lc,
                        adapt_end_time=self.lane.time_ + adapt_time,
                        target_lane=self.left_lane if lc_direction == -1 else self.right_lane,
                    ))

        self.gap_res_list = judge_res

    def lc_decision_making(self):
        if self.no_lc or self.lane_changing:
            return

        candidate_gap = []
        for gap_res in self.gap_res_list:
            if gap_res.target_acc is not None:
                lc_acc_benefit = 1 / (1 + np.exp(-gap_res.acc_gain))
                lc_route_desire = gap_res.route_gain
                lc_ttc_risk = min(np.exp(self.ttc_star - gap_res.ttc_risk), 1)

                decision = False
                lc_prob = self.fuzzy_logic.compute(lc_acc_benefit, lc_ttc_risk, self.rho)
                if lc_route_desire > 0.9 or lc_route_desire > np.random.uniform():
                    decision = True
                    lc_prob += lc_route_desire
                if not decision:
                    decision = lc_prob > np.random.uniform() and lc_prob > 0.5

                if decision:
                    candidate_gap.append((gap_res, lc_prob))

        if len(candidate_gap) == 0:
            return
        else:
            candidate_gap.sort(key=lambda x: x[1], reverse=True)
            gap_res, lc_prob = candidate_gap[0]
            self.lc_direction = gap_res.lc_direction

            if isinstance(self, Game_H_Vehicle):
                self.opti_gap = gap_res
                self.lane_changing = True

    def _no_car_correction(self, gap, lc_direction):
        """判别TR、TF、PC是否存在，若不存在则设置为不影响换道博弈的虚拟车辆
        :gap: 车辆间距
        """
        if lc_direction == 0:
            raise ValueError("lc_direction is 0, please check the code.")
        if gap == -1:  # 后向搜索
            if lc_direction == 1:
                TF = self.rr
                TR = self.rr.r if self.rr is not None else None
            else:
                TF = self.lr
                TR = self.lr.r if self.lr is not None else None
            if TF is None:
                return None, None, None, False
        elif gap == 1:  # 前向搜索
            if lc_direction == 1:
                TR = self.rf
                TF = self.rf.f if self.rf is not None else None
            else:
                TR = self.lf
                TF = self.lf.f if self.lf is not None else None
            if TR is None:
                return None, None, None, False
        elif gap == 0:  # 相邻搜索
            if lc_direction == 1:
                TF = self.rf
                TR = self.rr
            else:
                TF = self.lf
                TR = self.lr
        else:
            raise ValueError("gap should be -1, 0 or 1, please check the code.")

        if lc_direction == 1:
            lane = self.right_lane
        else:
            lane = self.left_lane

        if lane is None:
            return None, None, None, False

        PC = self.f
        if TR is None:
            position = self.position + np.array([-1e10, - lc_direction * self.lane.width])
            TR = Game_O_Vehicle(lane, self.type, - self.ID, self.length)
            TR.x = position[0]
            TR.y = position[1]
            TR.cf_model = self.cf_model
        if TF is None:
            position = self.position + np.array([1e10, - lc_direction * self.lane.width])
            TF = Game_O_Vehicle(lane, self.type, - self.ID, self.length)
            TF.x = position[0]
            TF.y = position[1]
            TF.cf_model = self.cf_model
        if PC is None:
            position = self.position + np.array([1e10, 0])
            PC = Game_O_Vehicle(lane, self.type, - self.ID, self.length)
            PC.x = position[0]
            PC.y = position[1]
            PC.cf_model = self.cf_model
        return TR, TF, PC, True


class Game_A_Vehicle(Game_Vehicle):
    """自动驾驶车辆"""

    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)
        self.TF_co = 1  # TF自动驾驶的公平协作程度 [0, 1]
        """TR, TF, PC"""

        self.LC_pred_traj_df: Optional[pd.DataFrame] = None
        """换道车对该车轨迹的预测"""

        self.N_MPC = 5
        self.mpc_solver: Optional[MPC_Solver] = None

        self.lc_step_length = -1  # 换道步数
        self.lc_end_step = -1  # 换道完成时的仿真步数
        self.lc_start_step = -1  # 换道开始时的仿真步数

        self.lc_time_max = 10  # 换道最大时间
        self.lc_time_min = 1  # 换道最小时间

        self.opti_game_res: Optional[GameRes] = None
        self.lc_traj: Optional[np.ndarray] = None
        self.is_game_leader = False

        self.last_cal_step = -1
        self.state = None

        self.re_cal_step = 5

    def lc_intention_judge(self):
        if self.lane == self.target_lane and self.is_gaming:
            # 恢复TR、TF的TIME_WANTED
            TR = self.opti_game_res.TR
            TR.clear_game_stra()
            TF = self.opti_game_res.TF
            TF.clear_game_stra()

            self.is_gaming = False
            self.is_game_leader = False
            self.lc_direction = 0
            self.opti_game_res = None
            self.opti_gap = None

            self.LC_CROSS_STEP = self.lane.step_

        # if self.lc_end_step < self.lane.step_:
        if (abs(self.y_c - self.target_lane.y_center) < 0.1
                and self.v_lat < 0.1 and abs(self.yaw) < 5 / 180 * np.pi):
            # 换道完成
            self.lane_changing = False
            self.lc_direction = 0
            self.LC_REAL_END_STEP = self.lane.step_

        if not self.lane_changing:
            super().lc_intention_judge()

    def lc_decision_making(self):
        """判断是否换道"""
        super().lc_decision_making()
        if self.no_lc:
            return
        # 根据上一时间步的TR行为策略更新rho_hat
        if self.is_gaming and self.is_game_leader:
            self.update_rho_hat()
        self.state = self.get_state_for_traj()
        if self.lane_changing or self.lc_direction != 0:
            # LC轨迹优化，如果博弈选择不换道，需要重新更新self.lane_changing，is_gaming以及target_lane
            if self.lane_changing and self.lane != self.target_lane and self.last_cal_step >= self.re_cal_step:
                self.stackel_berg()
            elif not self.lane_changing and self.lc_direction != 0:  # 初始博弈
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
            acc = min(acc, next_acc_block)
        else:
            # 横向控制
            acc, delta, is_end = self.mpc_solver.step_mpc()  # 默认计算出的符合约束的加速度和转向角
            if is_end:
                self.lane_changing = False
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
        self.lane_changing = False
        if remove_lc_direction:
            self.lc_direction = 0
        self.target_lane = self.lane
        self.lc_traj = None
        self.opti_game_res = None
        self.mpc_solver = None

    def update_rho_hat(self):
        """求解rho的范围，入股TR的rho_hat不在范围内，更新TR的rho_hat"""
        if not (self.is_game_leader and self.is_gaming) or self.last_cal_step != 1:
            return
        TR = self.opti_game_res.TR
        if isinstance(TR, Game_H_Vehicle):
            # 第一次估计TR的rho_hat默认为0.5
            rho_hat = self._get_rho_hat_s([TR])[0]
            TR_cost_lambda = self.opti_game_res.traj_data.TR_cost_lambda

            # 构建不等式求解问题
            rho_real = sp.symbols("rho_real")
            # 定义不等式
            inequality = sp.Ge(TR_cost_lambda(rho_real), TR_cost_lambda(rho_hat))  # f(x) >= g(x)
            # 求解不等式
            solution = sp.solve_univariate_inequality(inequality, rho_real, relational=False)
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
                    # if lower_bound <= self.rho_hat_s[TR.ID] <= upper_bound:
                    #     continue
                    # else:
                    self.rho_hat_s[TR.ID] = (lower_bound + upper_bound) / 2

    def stackel_berg(self):
        """基于stackelberg主从博弈理论的换道策略，计算得到参考轨迹（无论是否换道）"""
        # 计算不同策略（换道时间）的最优轨迹和对应效用值
        # cal_game_matrix需要更新周边车辆的策略，stackel_berg函数只更新自身的策略
        game_res_list = self.cal_game_matrix()
        # 获取最优策略
        # game_cost_list = [res.EV_cost if res.EV_cost is not None else np.inf for res in game_res_list]
        lc_game_res_list = [res for res in game_res_list
                            if res.EV_opti_series is not None and res.EV_opti_series["ego_stra"][1]]

        no_stra_found = False

        if len(lc_game_res_list) != 0:
            lc_game_cost_list = [res.EV_cost if res.EV_cost is not None else np.inf for res in lc_game_res_list]
            # 选择效用值最小的策略
            min_cost_idx = np.argmin(lc_game_cost_list)

            if lc_game_cost_list[min_cost_idx] != np.inf:
                opti_game_res = lc_game_res_list[min_cost_idx]
            else:
                no_lc_game_res_list = [res for res in game_res_list
                                       if res.EV_opti_series is not None and not res.EV_opti_series["ego_stra"][1]]
                no_lc_game_cost_list = [res.EV_cost if res.EV_cost is not None else np.inf
                                        for res in no_lc_game_res_list]
                min_cost_idx = np.argmin(no_lc_game_cost_list)
                opti_game_res = no_lc_game_res_list[min_cost_idx]
        else:
            no_lc_game_res_list = [res for res in game_res_list
                                   if res.EV_opti_series is not None and not res.EV_opti_series["ego_stra"][1]]
            no_lc_game_cost_list = [res.EV_cost if res.EV_cost is not None else np.inf
                                    for res in no_lc_game_res_list]
            if len(no_lc_game_cost_list) == 0:
                self.reset_lc_game()
                return
            else:
                min_cost_idx = np.argmin(no_lc_game_cost_list)
                opti_game_res = no_lc_game_res_list[min_cost_idx]

        # print([(game_res.EV_stra, game_res.EV_cost) for game_res in game_res_list])
        # min_cost_idx = np.argmin(game_cost_list)
        # opti_game_res = game_res_list[min_cost_idx]

        # 获取最优策略对应的轨迹
        opti_df = opti_game_res.EV_opti_series
        if opti_df is None:
            self.reset_lc_game()
            return

        # a0~a5, b0~b5, step, y1, ego_stra, TF_stra, TR_stra, PC_stra, ego_cost, TF_cost, TR_cost, PC_cost
        # 选择效用值最小的策略
        x_opt = opti_df.iloc[: 12].to_numpy().astype(float)
        T_opt, is_lc = opti_df.iloc[14]

        self.last_cal_step = 0

        if is_lc:
            self.reset_lc_game(remove_lc_direction=False)

            self.opti_game_res = opti_game_res
            self.opti_game_res.TR.set_game_stra(self.opti_game_res.TR_real_stra)
            if isinstance(self.opti_game_res.TF, Game_A_Vehicle):
                self.opti_game_res.TF.set_game_stra(self.opti_game_res.TF_stra)

            self.lc_step_length = np.round(T_opt / self.dt).astype(int)
            self.lc_start_step = self.lane.step_
            self.lc_end_step = self.lc_start_step + self.lc_step_length
            self.lc_traj = self.cal_ref_path(x_opt)

            self.lane_changing = True
            self.is_gaming = True
            self.target_lane = self.left_lane if self.lc_direction == -1 else self.right_lane
            self.game_time_wanted = self.time_wanted  # 不变
            self.is_game_leader = True
        else:
            # 如果博弈选择不换道，需要重新更新self.lane_changing，is_gaming以及target_lane
            self.reset_lc_game()

    def get_y_constraint(self, is_lc):
        lane_center_y = self.lane.y_center
        lane_width = self.lane.width

        if is_lc:
            y1 = lane_center_y + (- self.lc_direction) * lane_width
            y_limit_low = min(lane_center_y, y1) - lane_width / 2
            y_limit_up = max(lane_center_y, y1) + lane_width / 2
        else:
            y1 = lane_center_y
            y_limit_low = lane_center_y - lane_width / 2
            y_limit_up = lane_center_y + lane_width / 2

        return y1, y_limit_low, y_limit_up

    def cal_game_matrix(self):
        """backward induction method
        :return: opti_df (a0~a5, b0~b5, ego_stra, TF_stra, TR_stra, PC_stra,
         ego_cost, TF_cost, TR_cost, PC_cost)
        """
        game_res_list = []
        for gap in [-1, 0, 1]:
            TR, TF, PC, is_exist = self._no_car_correction(gap, self.lc_direction)
            if not is_exist:
                continue
            TR_real_stra, TF_stra, EV_stra = None, None, None
            TR_real_cost, TF_cost, EV_cost = None, None, None
            try:
                if isinstance(TR, Game_H_Vehicle) and isinstance(TF, Game_H_Vehicle):
                    # ego与TR的博弈
                    TR_strategy = TR.get_strategies(is_lc=False)
                    ego_strategy = self.get_strategies(is_lc=True)

                    cost_df, traj_data = self._cal_cost_df(
                        stra_product=itertools.product(ego_strategy, [None], TR_strategy),
                        cal_TF_cost=False, cal_TR_cost=True, TF=TF, TR=TR, PC=PC
                    )

                    # 根据估计的TR成本函数计算最小的当前车辆cost
                    # ego_stra包含了换道时间和换道方向，提取换道行为的cost_df_temp
                    cost_df_temp = cost_df[
                        (cost_df["ego_cost"] != np.inf) & (cost_df["ego_stra"].apply(lambda x: x[1]))]
                    if len(cost_df_temp) == 0:
                        game_res_list.append(GameRes(None, TF, TR, PC, None, None,
                                                     None, None, None, None, None))
                        continue
                    min_TR_cost_idx = cost_df_temp.groupby('ego_stra')['TR_cost_hat'].idxmin()  # 找到最小的TR_cost值
                    min_cost_idx = cost_df_temp.loc[min_TR_cost_idx]["ego_cost"].idxmin()  # 找到最小的cost值
                    EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                    EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
                    EV_opti_series = cost_df_temp.loc[min_cost_idx]

                    min_TR_cost_idx_real = cost_df_temp.groupby('ego_stra')['TR_cost_real'].idxmin()  # 找到最小的TR_cost值
                    min_cost_idx_real = cost_df_temp.loc[min_TR_cost_idx_real]["ego_cost"].idxmin()  # 找到最小的cost值
                    TR_real_stra = cost_df_temp.loc[min_cost_idx_real]["TR_stra"]
                    TR_real_cost = cost_df_temp.loc[min_cost_idx_real]["TR_cost_real"]

                    index = EV_opti_series["index"]
                    traj_data_opti = traj_data[index]

                    # print(cost_df[["ego_stra", "ego_cost"]])
                elif isinstance(TR, Game_H_Vehicle) and isinstance(TF, Game_A_Vehicle):
                    # ego与TR的博弈，ego与TF的合作
                    TR_strategy = TR.get_strategies(is_lc=False)
                    TF_strategy = TF.get_strategies(is_lc=False)
                    ego_strategy = self.get_strategies(is_lc=True)

                    cost_df, traj_data = self._cal_cost_df(
                        stra_product=itertools.product(ego_strategy, TF_strategy, TR_strategy),
                        cal_TF_cost=True, cal_TR_cost=True, TF=TF, TR=TR, PC=PC
                    )

                    cost_df["total_cost"] = cost_df["ego_cost"] + TF.TF_co * cost_df["TF_cost_real"]  # 合作效用
                    cost_df_temp = cost_df[
                        (cost_df["total_cost"] != np.inf) & (cost_df["ego_stra"].apply(lambda x: x[1]))]
                    if len(cost_df_temp) == 0:
                        game_res_list.append(GameRes(None, TF, TR, PC, None, None,
                                                     None, None, None, None, None))
                        continue
                    # 找到最小的TR_cost值
                    min_TR_cost_idx = cost_df_temp.groupby(by=["ego_stra", "TF_stra"])['total_cost'].idxmin()
                    min_cost_idx = cost_df_temp.loc[min_TR_cost_idx]["total_cost"].idxmin()  # 找到最小的cost值
                    TF_stra = cost_df_temp.loc[min_cost_idx]["TF_stra"]
                    EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                    TF_cost = cost_df_temp.loc[min_cost_idx]["TF_cost_real"]
                    EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
                    EV_opti_series = cost_df_temp.loc[min_cost_idx]

                    # 更新TR的策略
                    TR_real_stra = cost_df_temp.loc[min_cost_idx]["TR_stra"]
                    TR_real_cost = cost_df_temp.loc[min_cost_idx]["TR_real_cost"]

                    index = EV_opti_series["index"]
                    traj_data_opti = traj_data[index]
                elif isinstance(TR, Game_A_Vehicle) and isinstance(TF, Game_H_Vehicle):
                    # ego与TR的合作
                    TR_strategy = TR.get_strategies(is_lc=False)
                    ego_strategy = self.get_strategies(is_lc=True)

                    cost_df, traj_data = self._cal_cost_df(
                        stra_product=itertools.product(ego_strategy, [None], TR_strategy),
                        cal_TF_cost=False, cal_TR_cost=True, TF=TF, TR=TR, PC=PC
                    )

                    cost_df["total_cost"] = cost_df["ego_cost"] + TR.TF_co * cost_df["TR_cost_real"]  # 合作效用
                    cost_df_temp = cost_df[
                        (cost_df["total_cost"] != np.inf) & (cost_df["ego_stra"].apply(lambda x: x[1]))]
                    if len(cost_df_temp) == 0:
                        game_res_list.append(GameRes(None, TF, TR, PC, None, None,
                                                     None, None, None, None, None))
                        continue
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
                    TR_strategy = TR.get_strategies(is_lc=False)
                    TF_strategy = TF.get_strategies(is_lc=False)
                    ego_strategy = self.get_strategies(is_lc=True)

                    cost_df, traj_data = self._cal_cost_df(
                        stra_product=itertools.product(ego_strategy, TF_strategy, TR_strategy),
                        cal_TF_cost=True, cal_TR_cost=True, TF=TF, TR=TR, PC=PC
                    )

                    cost_df["total_cost"] = (
                            cost_df["cost"] + TR.TF_co * cost_df["TR_cost_real"] + TF.TF_co * cost_df["TF_cost_real"]
                    )
                    cost_df_temp = cost_df[
                        (cost_df["total_cost"] != np.inf) & (cost_df["ego_stra"].apply(lambda x: x[1]))]
                    if len(cost_df_temp) == 0:
                        game_res_list.append(GameRes(None, TF, TR, PC, None, None,
                                                     None, None, None, None, None))
                        continue
                    min_cost_idx = cost_df_temp["total_cost"].idxmin()  # 找到最小的cost值

                    TR_real_stra = cost_df_temp.loc[min_cost_idx]["TR_stra"]
                    TF_stra = cost_df_temp.loc[min_cost_idx]["TF_stra"]
                    EV_stra = cost_df_temp.loc[min_cost_idx]["ego_stra"]
                    TR_real_cost = cost_df_temp.loc[min_cost_idx]["TR_cost_real"]
                    TF_cost = cost_df_temp.loc[min_cost_idx]["TF_cost_real"]
                    EV_cost = cost_df_temp.loc[min_cost_idx]["ego_cost"]
                    EV_opti_series = cost_df_temp.loc[min_cost_idx]

                    index = EV_opti_series["index"]
                    traj_data_opti = traj_data[index]
                else:
                    print(f"车辆类型错误：TR: {type(TR)}, TF: {type(TF)}")
                    raise NotImplementedError("未知的车辆类型")
            except cvxpy.error.SolverError:
                return GameRes(None, TF, TR, PC, None, None, None, None, None, None, None)

            game_res_list.append(GameRes(
                cost_df, TF, TR, PC, EV_stra, TF_stra, TR_real_stra,
                EV_cost, TF_cost, TR_real_cost, EV_opti_series, traj_data_opti
            ))

        return game_res_list

    def _cal_cost_df(self, stra_product, cal_TF_cost, cal_TR_cost, TF, TR, PC):
        stra_product = list(stra_product)

        cost_data = []
        traj_data = []
        # a0~a5, b0~b5, step, y1, T, is_lc, TF_stra, TR_stra, cost, TF_cost, TR_cost
        for i, [ego_stra, TF_stra, TR_stra] in enumerate(stra_product):
            result, traj = self.cal_cost_given_stra(
                ego_stra, TF_stra, TR_stra,
                TF=TF, TR=TR, PC=PC,
                cal_TF_cost=cal_TF_cost, cal_TR_cost=cal_TR_cost
            )
            result.append(i)
            cost_data.append(result)
            traj_data.append(traj)

        cost_df = pd.DataFrame(
            cost_data,
            columns=[*[f"{i}{j}" for i in ["a", "b"] for j in range(6)], *["step", "y1"],
                     *["ego_stra", "TF_stra", "TR_stra", "TF_cost_hat", "TR_cost_hat",
                       'ego_cost', 'TF_cost_real', 'TR_cost_real', "index"]],
        )
        return cost_df, traj_data

    def cal_cost_given_stra(
            self, stra, TF_stra=None, TR_stra=None,
            TF: Game_Vehicle = None, TR: Game_Vehicle = None, PC: Game_Vehicle = None,
            cal_TF_cost=False, cal_TR_cost=False
    ):
        """根据策略计算效用值
        :return: a0~a5, b0~b5, step, y1, T, is_lc, TF_stra, TR_stra, TF_cost_hat, TR_cost_hat, cost,
         TF_cost_real, TR_cost_real,
        """
        T, is_lc = stra
        # 换道效用
        y1, y_limit_low, y_limit_up = self.get_y_constraint(is_lc=is_lc)
        (lc_result, [TF_traj, TR_traj, PC_traj, TF_PC_traj, TR_PC_traj, PC_PC_traj],
         [safe_cost, com_cost, eff_cost, route_cost]) = \
            self.opti_quintic_given_T(
                T, y1, y_limit_low, y_limit_up,
                TF, TR, PC,
                TF_stra, TR_stra,
                return_other_traj=True
            )

        x_opt = lc_result[1:13]
        times = np.arange(0, T + self.dt / 2, self.dt)
        if lc_result[0] != np.inf:
            EV_traj = np.vstack([np.array(get_xy_quintic(x_opt, t)) for t in times])
            # EV_cost = cal_other_cost(self, True, self.rho, EV_traj, [PC_traj, TR_traj, TF_traj])[0]
            # print("stra", stra, "\tTF_stra", TF_stra, "\tTR_stra", TR_stra,
            #       "\tsafe_cost", safe_cost, "\tcom_cost", com_cost, "eff_cost\t", eff_cost,
            #       "\troute_cost", route_cost, "\ttotal_cost", lc_result[0])
        else:
            EV_traj = None

        TF_rho_hat = self._get_rho_hat_s([TF])[0] if isinstance(TF, Game_H_Vehicle) else TF.rho
        TR_rho_hat = self._get_rho_hat_s([TR])[0] if isinstance(TR, Game_H_Vehicle) else TR.rho

        # TF
        TF_cost_hat, TF_real_cost, TF_cost_lambda = cal_other_cost(
            TF, cal_TF_cost, TF_rho_hat, TF_traj, [PC_traj])
        if is_lc:
            # TR
            TR_cost_hat, TR_real_cost, TR_cost_lambda = cal_other_cost(
                TR, cal_TR_cost, TR_rho_hat, TR_traj,
                [TF_traj, EV_traj] if EV_traj is not None else [TF_traj]
            )
        else:
            TR_cost_hat, TR_real_cost, TR_cost_lambda = cal_other_cost(
                TR, cal_TR_cost, TR_rho_hat, TR_traj,
                [TF_traj]
            )

        result = [*lc_result[1:], stra, TF_stra, TR_stra,
                  TF_cost_hat, TR_cost_hat,
                  lc_result[0], TF_real_cost, TR_real_cost]

        assert TR_cost_lambda is not None

        traj_data = TrajData(
            EV_traj, TF_traj, TR_traj, PC_traj,
            TF_PC_traj, TR_PC_traj, PC_PC_traj,
            TF_cost_lambda, TR_cost_lambda
        )
        return result, traj_data

    def opti_quintic_given_T(self, T, y1, y_limit_low, y_limit_up,
                             TF: Game_Vehicle = None, TR: Game_Vehicle = None, PC: Game_Vehicle = None,
                             TF_stra=None, TR_stra=None, return_other_traj=False):
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
        b_ineq3 = np.array([0] * len(A_ineq3))
        b_ineq4 = np.array([self.vel_desire] * len(A_ineq3))
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
        vehicles = [TF, TR, PC]
        (TF_traj, l_TF, TF_PC_traj), (TR_traj, l_TR, TR_PC_traj), (PC_traj, l_PC, PC_PC_traj) = \
            self.pred_traj_s(
                T,
                vehicles,
                stra_s=[TF_stra, TR_stra, None]
            )
        xt_TF, dxt_TF = TF_traj[:, 0], TF_traj[:, 1]
        xt_TR, dxt_TR = TR_traj[:, 0], TR_traj[:, 1]
        xt_PC, dxt_PC = PC_traj[:, 0], PC_traj[:, 1]

        if abs(y_limit_up - y_limit_low - self.lane.width) < 1e-3:
            # PC
            dxt = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps])
            A_ineq_7 = np.array([
                [1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])  # 纵向位置
            x_pos = A_ineq_7 @ x[: 6]  # 纵向位置
            v = dxt @ x[:6]
            PC_rear_x = xt_PC - l_PC
            b_ineq_7_part1 = PC_rear_x - v * self.time_safe
            # b_ineq_7_part1 = PC_rear_x - 2
            constraints += [x_pos <= b_ineq_7_part1]
            b_ineq_7_part2 = PC_rear_x - v * self.time_wanted
            # b_ineq_7_part2 = PC_rear_x - 2
            constraints += [x[12] >= x_pos - b_ineq_7_part2]

            # 安全性
            constraints += [x[13] == 0]
            constraints += [x[14] == 0]
            safe_cost = x[12]

            target_direction = 0
            step = np.nan
        else:
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
            # b_ineq_7_part1 = PC_rear_x - 2
            constraints += [x_pos_before <= b_ineq_7_part1]
            b_ineq_7_part2 = PC_rear_x - v * self.time_wanted
            # b_ineq_7_part2 = PC_rear_x - 2
            constraints += [x[12] >= x_pos_before - b_ineq_7_part2]

            # TF
            dxt_after = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in lc_after_times])
            v = dxt_after @ x[:6]
            TF_rear_x = xt_TF[step:] - l_TF
            b_ineq_8_part1 = TF_rear_x - v * self.time_safe
            # b_ineq_8_part1 = TF_rear_x - 2
            constraints += [x_pos_after <= b_ineq_8_part1]
            b_ineq_8_part2 = TF_rear_x - v * self.time_wanted
            # b_ineq_8_part2 = TF_rear_x - 2
            constraints += [x[13] >= x_pos_after - b_ineq_8_part2]

            # TR
            TR_head_x = xt_TR[step:] + l_TR
            TR_v = dxt_TR[step:]
            b_ineq_9_part1 = TR_head_x + self.time_safe * TR_v
            # b_ineq_9_part1 = TR_head_x + 2
            constraints += [x_pos_after >= b_ineq_9_part1]
            b_ineq_9_part2 = TR_head_x + self.time_wanted * TR_v
            # b_ineq_9_part2 = TR_head_x + 2
            constraints += [x[14] >= b_ineq_9_part2 - x_pos_after]

            # 安全性
            safe_cost = cvxpy.max(x[12: 15])

            target_direction = -1 if y1 > (y_limit_low + y_limit_up) / 2 else 1

        _, route_cost = self.cal_route_cost(self.lane.index + target_direction)
        _, route_ori = self.cal_route_cost(self.lane.index)
        route_cost = route_cost - route_ori
        # print(T, target_direction, route_gain)

        # 舒适性：横纵向jerk目标
        k_c = 0.5
        A_xy_acc = np.array([
            [0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3] for t in time_steps
        ])
        # com_cost = 0
        com_cost = k_c * cvxpy.max(cvxpy.abs(A_xy_acc @ x[:6]))
        com_cost += k_c * cvxpy.max(cvxpy.abs(A_xy_acc @ x[6:12]))

        # 效率
        A_dxy = np.array([
            [0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps
        ])
        vx = A_dxy @ x[:6]
        eff_cost = cvxpy.max(vx[0] - cvxpy.mean(vx))
        rho_ = self.rho
        cost = rho_ * (safe_cost + com_cost) + (1 - rho_) * eff_cost + route_cost

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

        if return_other_traj:
            other_traj = [TF_traj, TR_traj, PC_traj, TF_PC_traj, TR_PC_traj, PC_PC_traj]
            return result, other_traj, [safe_cost.value, com_cost.value, eff_cost.value, route_cost]
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

    def get_strategies(self, is_lc: bool = False):
        return [self.time_wanted]

    def pred_self_traj(self, time_len, stra=None,
                       target_lane: "LaneAbstract" = None,
                       PC_traj=None, to_ndarray=True):
        step_num = round(time_len / self.dt) + 1
        if to_ndarray:
            traj_list = [self.get_traj_point().to_ndarray()] * step_num
        else:
            traj_list = [self.get_traj_point()] * step_num
        return np.array(traj_list), None
