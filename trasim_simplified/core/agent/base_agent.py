# -*- coding: utf-8 -*-
# @Time : 2025/5/8 21:59
# @Author : yzbyx
# @File : base_agent.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.agent import Vehicle
from trasim_simplified.core.agent.collision_risk import calculate_collision_risk
from trasim_simplified.core.agent.fuzzy_logic import FuzzyLogic
from trasim_simplified.core.agent.utils import interval_intersection
from trasim_simplified.core.constant import V_TYPE, RouteType, GapJudge, VehSurr, TrajPoint, StraInfo, LcGap
from trasim_simplified.core.kinematics.lcm import LCModel_Mobil

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class Base_Agent(Vehicle):
    fuzzy_logic = FuzzyLogic()

    def __init__(self, lane: 'LaneAbstract', type_: V_TYPE, id_: int, length: float):
        super().__init__(lane, type_, id_, length)

    def pred_lc_risk(self, time_len=3., lc_direction=None):
        """车辆当前时刻换道的风险/不换道的风险TTC"""
        stra_info = StraInfo(self, time_len, 1, 0)

        if lc_direction == -1:
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
                stra_info.lc_direction = lc_direction
                stra_info.lc_gap = LcGap(self.lf, self.lr, self.leader)
                traj_ev_left, _ = self.pred_self_traj(stra_info, to_ndarray=False)

                ev_lp_ttc_2d = calculate_collision_risk(traj_ev_left, traj_lp) if traj_lp is not None else np.inf
                ev_lr_ttc_2d = calculate_collision_risk(traj_ev_left, traj_lr) if traj_lr is not None else np.inf
                ev_cp_ttc_2d = calculate_collision_risk(traj_ev_left, traj_cp) if traj_cp is not None else np.inf
                min_ttc_2d_left = min(np.min(ev_lp_ttc_2d), np.min(ev_lr_ttc_2d), np.min(ev_cp_ttc_2d))

            return min_ttc_2d_left

        if lc_direction == 1:
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
                stra_info.lc_direction = lc_direction
                stra_info.lc_gap = LcGap(self.rf, self.rr, self.leader)
                traj_ev_right, _ = self.pred_self_traj(stra_info, to_ndarray=False)

                ev_rp_ttc_2d = calculate_collision_risk(traj_ev_right, traj_rp) if traj_rp is not None else np.inf
                ev_rr_ttc_2d = calculate_collision_risk(traj_ev_right, traj_rr) if traj_rr is not None else np.inf
                ev_cp_ttc_2d = calculate_collision_risk(traj_ev_right, traj_cp) if traj_cp is not None else np.inf
                min_ttc_2d_right = min(np.min(ev_rp_ttc_2d), np.min(ev_rr_ttc_2d), np.min(ev_cp_ttc_2d))

            return min_ttc_2d_right

        if lc_direction == 0:
            traj_cp = self.pred_net.pred_traj(self.leader.pack_veh_surr(), type_="net", time_len=time_len) \
                if self.leader is not None else None
            traj_ev_stay, _ = self.pred_self_traj(stra_info, to_ndarray=False, ache=False)
            ev_cp_ttc_2d = calculate_collision_risk(traj_ev_stay, traj_cp) if traj_cp is not None else np.inf
            min_ev_cp_ttc_2d = np.min(ev_cp_ttc_2d)
            return min_ev_cp_ttc_2d
        return None

    def _cal_lane_cost(self, lane_index, x):
        # 目标路径
        target_direction = 0
        weaving_length = self.lane.road.end_weaving_pos - self.lane.road.start_weaving_pos
        d_r = self.lane.road.end_weaving_pos - x
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

        self.risk_2d = self.pred_lc_risk(lc_direction=self.lc_direction, time_len=0)

        if self.lane_changing and self.risk_2d >= self.ttc_star:
            return
        # if self.opti_gap is not None and self.is_keep_lane_center():
        #     self.reset_lc_state()
        if self.lane_changing and self.risk_2d < self.ttc_star:
            self.reset_lc_state()

        judge_res = []
        for adapt_time in np.arange(0, 3.1, 1):
            for lc_direction in [-1, 1]:
                for gap in [-1, 0, 1]:
                    if self.lane_can_lc is not None and (self.lane.index + lc_direction not in self.lane_can_lc):
                        continue
                    if self.right_lane is None and lc_direction == 1 or self.no_right_lc:
                        continue
                    if self.left_lane is None and lc_direction == -1 or self.no_left_lc:
                        continue

                    target_lane_index = self.lane.index + lc_direction
                    if (self.route_type == RouteType.mainline and
                            target_lane_index not in self.destination_lane_indexes):
                        continue
                    if self.route_type == RouteType.merge and lc_direction != -1:
                        continue
                    if self.route_type == RouteType.diverge and lc_direction != 1:
                        continue

                    TR, TF, PC, CR = self._no_car_correction(gap, lc_direction)
                    TR_traj = self.pred_net.pred_traj(TR.pack_veh_surr(), time_len=adapt_time)
                    TF_traj = self.pred_net.pred_traj(TF.pack_veh_surr(), time_len=adapt_time)
                    PC_traj = self.pred_net.pred_traj(PC.pack_veh_surr(), time_len=adapt_time)

                    is_feasible, target_acc = self._lc_target_acc(
                        adapt_time, PC, TR, TF, PC_traj, TR_traj, TF_traj
                    )
                    if not is_feasible:
                        continue

                    is_acc_ok, acc_gain = self._lc_acc_benefit(
                        target_acc, adapt_time, PC, TR, TF, PC_traj, TR_traj, TF_traj
                    )

                    is_ttc_ok, ttc_risk = self._lc_safe_risk(
                        target_acc, adapt_time, PC, TR, TF, PC_traj, TR_traj, TF_traj
                    )

                    # 目标路径激励
                    route_incentive = self.route_incentive(lc_direction)

                    # 队列换道激励
                    platoon_incentive = self._platoon_incentive(lc_direction)

                    # 交织换道激励
                    weaving_incentive = self._weaving_incentive(lc_direction)

                    # 换道概率意图
                    lc_base_prob = self.fuzzy_logic.compute(acc_gain, ttc_risk, self.rho)

                    # 换道意图
                    if is_ttc_ok:  # ATTENTION：未考虑is_acc_ok
                        lc_prob = lc_base_prob + route_incentive + platoon_incentive + weaving_incentive
                    else:
                        lc_prob = 0

                    judge_res.append(GapJudge(
                        self.lane.step_,
                        lc_direction, gap, target_acc, adapt_time,
                        self, TF=TF, TR=TR, PC=PC,

                        acc_gain=acc_gain,
                        ttc_risk=ttc_risk,
                        route_gain=route_incentive,
                        platoon_gain=platoon_incentive,
                        weaving_gain=weaving_incentive,
                        lc_prob=lc_prob,

                        adapt_end_time=self.lane.time_ + adapt_time,
                        target_lane=self.left_lane if lc_direction == -1 else self.right_lane,
                    ))

        self.gap_res_list: Optional[list[GapJudge]] = judge_res

    def _lc_target_acc(self, adapt_time, PC, TR, TF, PC_traj, TR_traj, TF_traj):
        target_acc = np.nan

        TR_end = TR_traj[-1]
        TF_end = TF_traj[-1]
        # 判断速度调整时间内能否以舒适的加减速度达到可行换道位置，用于MOBIL模型判断
        # x_safe_min = TR_end.x + self.length + self.time_safe * TR_end.vx
        # x_safe_max = TF_end.x - TF.length - self.time_safe * TF_end.vx
        x_safe_min = TR_end.x + self.length + self.safe_s0
        x_safe_max = TF_end.x - TF.length - self.safe_s0

        if x_safe_max < x_safe_min:
            return False, target_acc

        if adapt_time == 0:
            return False, target_acc
        acc_1 = 2 * (x_safe_max - self.x - self.v * adapt_time) / adapt_time ** 2
        acc_2 = 2 * (x_safe_min - self.x - self.v * adapt_time) / adapt_time ** 2
        # 若(acc_1, acc_2)与[self.acc_max, - self.dec_max]的交集不为空，
        # 则选择加速度绝对值最小的加速度作为调整加速度
        acc_ = np.array([acc_2, acc_1])
        acc = interval_intersection(acc_, (- self.dec_max, self.acc_max))
        if acc is None or acc[0] == acc[1]:
            return False, target_acc

        if acc[0] <= 0 <= acc[1]:
            target_acc = 0
        elif acc[0] > 0:
            target_acc = acc[0]
        else:
            target_acc = acc[1]

        return True, target_acc

    def _lc_acc_benefit(self, target_acc, adapt_time, PC, TR, TF, PC_traj, TR_traj, TF_traj):
        TR_end = TR_traj[-1]
        TF_end = TF_traj[-1]
        PC_end = PC_traj[-1]

        v_f = self.v + target_acc * adapt_time
        x_f = self.x + self.v * adapt_time + 0.5 * target_acc * adapt_time ** 2

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

        veh_surr = VehSurr(ego_f, cp=PC_f, rp=TF_f, rr=TR_f)
        is_acc_ok, acc_gain = LCModel_Mobil.mobil(
            1, veh_surr,
            return_acc_gain=True,
            LANE_CHANGE_MAX_BRAKING_IMPOSED=self.dec_max,
            LANE_CHANGE_MIN_ACC_GAIN=-np.inf,
            POLITENESS=self.rho
        )

        acc_gain = max(min(acc_gain / 3, 1), 0)

        return is_acc_ok, acc_gain

    def _lc_safe_risk(self, target_acc, adapt_time, PC, TR, TF, PC_traj, TR_traj, TF_traj):
        is_safe_ok = True

        ego_traj = [TrajPoint(
            x=self.x + self.v * t + 0.5 * target_acc * (t ** 2),
            y=self.y + self.v_lat * t, speed=self.v + target_acc * t,
            yaw=self.yaw, length=self.length, acc=target_acc,
            width=self.width
        ) for t in np.arange(0, adapt_time + self.dt / 2, self.dt)]

        risk_pc = float(np.min(np.array(calculate_collision_risk(ego_traj, PC_traj))))
        risk_tr = float(np.min(np.array(calculate_collision_risk(ego_traj, TR_traj))))
        risk_total = min(risk_pc, risk_tr)

        if risk_total < self.ttc_star:
            is_safe_ok = False

        ttc_risk = min(np.exp(self.ttc_star - risk_pc), 1)
        return is_safe_ok, ttc_risk

    def lc_decision_making(self, set_lane_changing=True):
        if self.no_lc or self.lane_changing:
            return

        final_candidate_gap = []
        for gap_res in self.gap_res_list:
            lc_prob = gap_res.lc_prob
            if lc_prob > np.random.uniform():
                final_candidate_gap.append((gap_res, lc_prob))

        self.lc_hold_time += self.dt
        if len(final_candidate_gap) == 0 or self.lc_hold_time < np.random.random() / self.decision_frequency:
            return
        self.lc_hold_time = 0

        final_candidate_gap.sort(key=lambda x: x[1], reverse=True)
        gap_res, lc_prob = final_candidate_gap[0]
        self.lc_direction = gap_res.lc_direction
        self.opti_gap = gap_res
        if set_lane_changing:
            self.target_lane = gap_res.target_lane
            self.lane_changing = True
            self.lc_cross = False

    def _weaving_incentive(self, lc_direction):
        if self.lane.index in self.destination_lane_indexes:
            return 0
        tp = self.lf if lc_direction == -1 else self.rf
        if tp is not None and tp.lane_changing and not tp.lc_cross:
            dhw = tp.x - self.x
            if tp.lc_direction == - lc_direction:
                return self.weaving_incentive * np.exp(- dhw / self.incentive_range)
        return 0

    def _platoon_incentive(self, lc_direction):
        if self.lane.index in self.destination_lane_indexes:
            return 0
        if self.leader is not None and self.leader.lane_changing and not self.leader.lc_cross:
            dhw = self.leader.x - self.x
            if self.leader.lc_direction == lc_direction:
                return self.platoon_incentive * np.exp(- dhw / self.incentive_range)
        return 0

    def route_incentive(self, lc_direction):
        _, e_r_stay = self._cal_lane_cost(self.lane.index, self.x)
        _, e_r_lc = self._cal_lane_cost(self.lane.index + lc_direction, self.x)
        return e_r_stay - e_r_lc

    def _no_car_correction(self, gap, lc_direction, return_RR=False):
        """判别TR, TF, PC, CR是否存在，若不存在则设置为不影响换道博弈的虚拟车辆
        :gap: 车辆间距
        """
        if lc_direction == -1 and self.left_lane is None:
            raise ValueError("left lane is None, please check the code.")
        if lc_direction == 1 and self.right_lane is None:
            raise ValueError("right lane is None, please check the code.")
        if lc_direction == 0:
            TF = TR = None
        else:
            if gap == -1:  # 后向搜索
                if lc_direction == 1:
                    TF = self.rr
                    TR = self.rr.r if self.rr is not None else None
                else:
                    TF = self.lr
                    TR = self.lr.r if self.lr is not None else None
            elif gap == 1:  # 前向搜索
                if lc_direction == 1:
                    TR = self.rf
                    TF = self.rf.f if self.rf is not None else None
                else:
                    TR = self.lf
                    TF = self.lf.f if self.lf is not None else None
            elif gap == 0:  # 相邻搜索
                if lc_direction == 1:
                    TF = self.rf
                    TR = self.rr
                else:
                    TF = self.lf
                    TR = self.lr
            else:
                raise ValueError("gap should be -1, 0 or 1, please check the code.")

        if lc_direction == -1:
            lane = self.left_lane
        elif lc_direction == 1:
            lane = self.right_lane
        else:
            lane = self.lane

        PC = self.f
        CR = self.r
        if TR is None:
            position = self.position + np.array([-1e10, - lc_direction * self.lane.width])
            TR = self._make_dummy_agent(lane, self.type, -self.ID - 1, self.length, position[0], position[1])
        if TF is None:
            position = self.position + np.array([1e10, - lc_direction * self.lane.width])
            TF = self._make_dummy_agent(lane, self.type, -self.ID - 2, self.length, position[0], position[1])
        if PC is None:
            position = self.position + np.array([1e10, 0])
            PC = self._make_dummy_agent(lane, self.type, -self.ID - 3, self.length, position[0], position[1])
        if CR is None:
            position = self.position + np.array([-1e10, 0])
            CR = self._make_dummy_agent(lane, self.type, -self.ID - 4, self.length, position[0], position[1])

        if return_RR:
            if TR.r is None:
                TRR = self._make_dummy_agent(TR.lane, TR.type, -TR.ID - 5, TR.length, TR.x * 2, TR.y)
            else:
                TRR = TR.r
            if CR.r is None:
                CRR = self._make_dummy_agent(CR.lane, CR.type, -CR.ID - 6, CR.length, CR.x * 2, CR.y)
            else:
                CRR = CR.r
            if TF.f is None:
                TFF = self._make_dummy_agent(TF.lane, TF.type, -TF.ID - 7, TF.length, TF.x * 2, TF.y)
            else:
                TFF = TF.f
            if PC.f is None:
                CPP = self._make_dummy_agent(PC.lane, PC.type, -PC.ID - 8, PC.length, PC.x * 2, PC.y)
            else:
                CPP = PC.f

            return TR, TF, PC, CR, TRR, CRR, TFF, CPP
        # TR.f = TF
        # CR.f = self

        return TR, TF, PC, CR

    def _make_dummy_agent(self, lane, type_, id_, length, x, y):
        """创建一个虚拟车辆"""
        dummy_agent = Base_Agent(lane, type_, id_, length)
        dummy_agent.x = x
        dummy_agent.y = y
        dummy_agent.speed = self.speed
        dummy_agent.cf_model = self.cf_model
        return dummy_agent

    def get_lane_indexes(self, y_s):
        """获取轨迹所在车道的索引"""
        lane_indexes = []
        for y in y_s:
            for lane in self.lane.road.lane_list:
                if lane.y_right < y <= lane.y_left:
                    lane_indexes.append(lane.index)
                    break
            else:
                lane_indexes.append(np.nan)
        return np.array(lane_indexes)
