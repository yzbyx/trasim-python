# -*- coding: utf-8 -*-
# @Time : 2025/3/22 22:13
# @Author : yzbyx
# @File : game_agent.py
# Software: PyCharm
import abc
import itertools
from abc import ABC
from typing import Optional, Union

import cvxpy
import numpy as np
import pandas as pd

from trasim_simplified.core.agent.agent import AgentBase
from trasim_simplified.core.agent.utils import get_xy_quintic, get_y_guess


class Game_Vehicle(AgentBase, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lf: Optional['Game_Vehicle'] = None
        self.lr: Optional['Game_Vehicle'] = None
        self.f: Optional['Game_Vehicle'] = None
        self.r: Optional['Game_Vehicle'] = None
        self.rf: Optional['Game_Vehicle'] = None
        self.rr: Optional['Game_Vehicle'] = None

        self.T_b = 1.5
        """临界期望时距"""
        self.rho = 0.5
        """激进系数（仅用于HV）"""

        self.lc_acc_gain = []
        """换道加速度增益"""

        self.T_safe = 0.5
        self.T_desire = 1

    @abc.abstractmethod
    def cal_cost_by_traj(self, traj, PC_traj=None, LC_traj=None, rho=None, return_lambda=False):
        """只有不换道车辆才会使用此方法"""
        pass

    @abc.abstractmethod
    def get_strategies(self, is_lc: bool = False):
        """获取车辆策略"""
        pass

    @abc.abstractmethod
    def pred_self_traj(self, stra, T, is_lc: bool = False, uniform_speed=False):
        """在策略下预测自车轨迹"""
        pass

    def update_surrounding_vehicle(self):
        self.f = self.leader
        self.r = self.follower
        self.lr, self.lf, self.rr, self.rf = self.lane.road.get_neighbour_vehicles(self, "all")


class Game_A_Vehicle(Game_Vehicle):
    """自动驾驶车辆"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.TF_co = 1  # TF自动驾驶的公平协作程度 [0, 1]
        self.rho_hat_s = {}  # 储存车辆id_对应的rho_hat
        self.combine_vehicles: list[Game_Vehicle] = []
        """TR, TF, PC"""
        self.cost_df = None  # 存储博弈策略与对应效用值
        self.opti_idx = None  # 存储最优策略的索引

        self.LC_pred_traj_df: Optional[pd.DataFrame] = None
        """换道车对该车轨迹的预测"""

        self.N_MPC = 20
        self.ref_path = None


    def act(self, action: Union[dict, str] = None):
        if self.crashed:
            return
        self.update_rho_hat()  # 根据上一时间步的TR行为策略更新rho_hat

        self.state = self.get_state()
        # 评估换道
        self.lane_change_judge()

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
            acc = self.cf_longitude_control()
        else:
            self.lane_changing = True
            # 横向控制
            opti_v, opti_delta = self.lc_lateral_control()
            delta, v = opti_delta[0], opti_v[0]
            # 纵向控制
            acc = self.lc_longitude_control(v)
            self.lc_step_conti += 1

        Vehicle.act(
            self, {"acceleration": acc, "steering": delta}
        )

    def update_rho_hat(self):
        """求解rho的范围，入股TR的rho_hat不在范围内，更新TR的rho_hat"""
        # TODO update_rho_hat
        if self.combine_vehicles is None or (self.lane_changing and self.lane.step_ >= self.lc_entered_step):
            return
        if self.cost_df is None:
            return
        if len(self.combine_vehicles) == 0:
            return
        TR = self.combine_vehicles[0]
        if isinstance(TR, Game_H_Vehicle):
            if self.rho_hat_s.get(TR.id_, None) is None:
                # 第一次估计TR的rho_hat默认为0.5
                self.rho_hat_s[TR.id_] = 0.5
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

    def lane_change_judge(self):
        """判断是否换道并计算换道轨迹"""
        self.update_surrounding_vehicle()
        # 处于跟驰状态或有换道中有碰撞风险，重新计算目标车道
        if not self.lane_changing or self.pred_risk():
            # 计算是否有换道需求，更新self.lane_changing以及self.target_lane_index
            self.cal_lc_direction()
            if self.lane_changing or self.lc_direction != 0:
                # LC轨迹优化，如果博弈选择不换道，需要重新更新self.lane_changing以及self.target_lane_index
                self.stackel_berg()
                # 计算参考轨迹
                if self.lc_direction != 0:
                    self.cal_ref_path()

    def cal_ref_path(self):
        times = np.arange(0, (self.lc_step_length + 0.5) * self.dt, self.dt)
        # x, dx, ddx, y, dy, ddy
        self.ref_path = np.vstack([np.array(get_xy_quintic(self.lc_x_opti, t)) for t in times])
        x, dx, ddx, y, dy, ddy = list(self.lc_end_state)
        times_after = np.arange(self.dt, (self.N_MPC + 0.5) * self.dt, self.dt).reshape(-1, 1)
        self.ref_path = np.vstack([self.ref_path,
                                   np.hstack([
                                       x + dx * times_after + 0.5 * ddx * times_after ** 2,
                                       np.tile(dx, (self.N_MPC, 1)),
                                       np.tile(ddx, (self.N_MPC, 1)),
                                       np.tile(y, (self.N_MPC, 1)),
                                       np.tile(dy, (self.N_MPC, 1)),
                                       np.tile(ddy, (self.N_MPC, 1))
                                   ])])  # 填补参考轨迹

        # 使用差分的方式计算路径点的一阶导和二阶导，从而得到切线方向和曲率
        dx = self.ref_path[:, 1]
        dy = self.ref_path[:, 4]
        ddx = self.ref_path[:, 2]
        ddy = self.ref_path[:, 5]

        yaw = np.arctan2(dy, dx).reshape(-1, 1)  # yaw
        # 计算曲率:设曲线r(t) =(x(t),y(t)),则曲率k=(x'y" - x"y')/((x')^2 + (y')^2)^(3/2).
        # 参考：https://blog.csdn.net/weixin_46627433/article/details/123403726
        k = ((ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))).reshape(-1, 1)  # 曲率k计算
        # 参考速度
        v = (np.linalg.norm(np.vstack((dx, dy)), axis=0)).reshape(-1, 1)

        self.ref_path = np.hstack([self.ref_path, yaw, k, v])

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
            self.lc_start_step = self.road.step_
            self.lc_end_step = self.lc_start_step + self.lc_step_length
            self.lc_entered_step = self.lc_start_step + step_opt
            self.lc_end_state = get_xy_quintic(x_opt, T_opt)
            self.lc_start_state = self.get_state()
            self.lc_x_opti = x_opt
        else:
            self.lc_direction = 0
            self.target_lane_index = self.lane_index

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
        :return: opti_df (a0~a5, b0~b5, ego_stra, TF_stra, TR_stra, PC_stra, ego_cost, TF_cost, TR_cost, PC_cost)
        """
        TR, TF, PC = (self.lr, self.lf, self.f) if self.lc_direction == -1 else (self.r, self.lf, self.r)
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

                # 根据真实的TR成本函数计算真正的TR_stra
                df_temp = cost_df[cost_df["ego_stra"] == cost_df.loc[min_cost_idx]["ego_stra"]]
                min_TR_real_cost_idx = df_temp["TR_real_cost"].idxmin()

                # 更新TR的策略
                TR.TIME_WANTED = df_temp.loc[min_TR_real_cost_idx]["TR_stra"]
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

                cost_df["total_cost"] = cost_df["cost"] + TF.TF_co * cost_df["TF_cost"]  # 合作效用
                cost_df_temp = cost_df[cost_df["ego_cost"] != np.inf]
                min_TR_cost_idx = cost_df_temp.groupby(by=["ego_stra", "TF_stra"])['TR_cost'].idxmin()  # 找到最小的TR_cost值
                min_cost_idx = cost_df.loc[min_TR_cost_idx]["total_cost"].idxmin()  # 找到最小的cost值

                temp_df = cost_df[(cost_df["ego_stra"] == cost_df.loc[min_cost_idx]["ego_stra"]) &
                                  (cost_df["TF_stra"] == cost_df.loc[min_cost_idx]["TF_stra"]) &
                                  (cost_df["total_cost"] != np.inf)]
                min_TR_real_cost_idx = temp_df["TR_real_cost"].idxmin()

                # 更新TR的策略
                TR.TIME_WANTED = temp_df.loc[min_TR_real_cost_idx]["TR_stra"]
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
        except Exception:
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
                       for name in ["ego", "TF", "TR", "PC"]]]
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
        cost, traj = lc_result[0], np.vstack([np.array(get_xy_quintic(lc_result[1:13], t)) for t in time_steps])

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
            TF_traj, PC_traj=TF_PC_traj, LC_traj=traj, rho=TF.rho) if cal_TF_cost else np.nan

        # TR
        TR_cost = TR.cal_cost_by_traj(
            TR_traj, PC_traj=TR_PC_traj, LC_traj=traj, rho=TR_rho) if cal_TR_cost else np.nan
        TR_real_cost = TR_cost if isinstance(TR, Game_A_Vehicle) else (TR.cal_cost_by_traj(
            TR_traj, PC_traj=TR_PC_traj, LC_traj=traj, rho=TR.rho) if cal_TR_cost else np.nan)

        # PC
        PC_cost = PC.cal_cost_by_traj(
            PC_traj, PC_traj=PC_PC_traj, rho=PC_rho) if cal_PC_cost else np.nan
        PC_real_cost = PC_cost if isinstance(PC, Game_A_Vehicle) else PC.cal_cost_by_traj(
            PC_traj, PC_traj=PC_PC_traj, LC_traj=traj, rho=PC.rho) if cal_PC_cost else np.nan

        result = np.hstack([lc_result[1:], [TF_stra, TR_stra, PC_stra],
                            [cost, TF_cost, TR_cost, PC_cost,
                             np.nan, TF_real_cost, TR_real_cost, PC_real_cost]]).reshape(1, -1)
        return result

    def _get_rho_hat_s(self, vehicles):
        rho_hat_s = []
        for v in vehicles:
            if v is not None:
                rho_hat_s.append(self.rho_hat_s.get(v.id_, 0.5))
            else:
                rho_hat_s.append(None)
        return rho_hat_s

    def cal_cost_by_traj(self, traj, PC_traj=None, LC_traj=None, rho=None, return_lambda=False):
        """只有不换道车辆才会使用此方法"""
        k_sf = 2
        k_c = 10
        k_e = 0.4
        if LC_traj is not None:
            # 安全性计算
            assert len(traj) == len(LC_traj), "The length of traj is not equal to the len(LC_traj)."
            x_f = traj[-1, 0]
            v_f = traj[-1, 1]
            x_LC_f = LC_traj[-1, 0]
            safe_cost = k_sf * (x_LC_f - x_f - self.T_desire * v_f) ** 2
        else:
            # 安全性计算
            assert len(traj) == len(PC_traj), "The length of traj is not equal to the len(PC_traj)."
            x_f = traj[-1, 0]
            v_f = traj[-1, 1]
            x_PC_f = PC_traj[-1, 0]
            safe_cost = k_sf * (x_PC_f - x_f - self.T_desire * v_f) ** 2
        # 舒适性计算
        com_cost = k_c * np.sum(np.power(np.diff(traj[:, 2]) / self.dt, 2)) * self.dt
        # 效率计算
        eff_cost = k_e * np.sum(np.power(traj[:, 1] - self.target_speed, 2)) * self.dt

        def cost(rho_):
            return safe_cost + com_cost + eff_cost

        if return_lambda:
            return cost

        return cost(None)

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
                traj, PC_traj = self.pred_self_traj_IDM(self.f, T, PC_traj=None)
                self.TIME_WANTED = ori_T_want
        return traj, PC_traj

    def get_strategies(self, is_lc: bool = False):
        if is_lc:
            # 换道时间
            strategy = list(np.arange(max(self.dt, self.lc_time_min - self.lc_step_conti * self.dt),
                                      max(self.dt, self.lc_time_max), self.dt * 10))  # ATTENTION: 10倍步长
        else:
            # 期望时距
            strategy = list(np.arange(0.5, 5, 0.5))
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
        ([TF_traj, dec_max_TF, l_TF, TF_PC_traj],
         [TR_traj, dec_max_TR, l_TR, TR_PC_traj],  # FIXME: 此处的TR预测轨迹假设不受ego的换道影响
         [PC_traj, dec_max_PC, l_PC, PC_PC_traj]) = (
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
        step = np.where((np.diff((Y <= (y_limit_low + y_limit_up) / 2)) != 0))[0][0] + 1 # 进入目标车道的初始时刻
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

    def pred_traj(self, T, vehicles: list[Game_Vehicle], stra_s) -> list[tuple[np.ndarray, float, float, np.ndarray]]:
        # TODO pred_traj
        pred_traj_s = []
        for v, stra in zip(vehicles, stra_s):
            v = v if v is not None else self.f if self.lc_direction == 1 else self.r
            dec_max, l = v.dec_max, v.length
            traj, PC_traj = v.pred_self_traj(stra, T, is_lc=False, uniform_speed=False)
            pred_traj_s.append((traj, dec_max, l, PC_traj))
        return pred_traj_s


class Game_H_Vehicle(Game_Vehicle):
    def __init__(
            self,
            road: Road,
            position: Vector,
            heading: float = 0,
            speed: float = 0,
            target_speed: float = None,
            route: Route = None,
            speed_control_type: str = None,
            dt: float = 0.05,
            N_MPC: int = 20,
            Np: int = 20,
            T_safe=0.5
    ):
        super().__init__(
            road, position, heading, speed, target_speed, route, speed_control_type, dt, N_MPC, Np, T_safe
        )
        self.cost_lambda = None

    def cal_cost_by_traj(self, traj, PC_traj=None, LC_traj=None, rho=None, return_lambda=False):
        """只有不换道车辆才会使用此方法"""
        assert PC_traj is not None, "The PC_traj is None."
        if LC_traj is not None:  # 说明换道轨迹已经计算
            assert len(traj) == len(LC_traj), "The length of traj and LC_traj is not equal."
            # 安全性评估
            x_LC_f = LC_traj[-1, 0]
            x_f = traj[-1, 0]
            v_f = traj[-1, 1]
            thw = (x_LC_f - x_f) / v_f
            cost_safe = max(0, 1 - thw / self.T_b)
        else:
            assert len(traj) == len(PC_traj), "The length of traj and PC_traj is not equal."
            x_PC_f = PC_traj[-1, 0]
            x_f = traj[-1, 0]
            v_f = traj[-1, 1]
            thw = (x_PC_f - x_f) / v_f
            cost_safe = max(0, 1 - thw / self.T_b)
        # 空间优势评估（以PC为基准）
        x_f = traj[-1, 0]
        v_f = traj[-1, 1]
        x_PC_f = PC_traj[0, 0] + PC_traj[0, 1] * PC_traj.shape[0] * self.dt  # 不协同的情况下，PC的预测轨迹
        cost_space = max(0, (x_PC_f - x_f) / (v_f * self.T_b))
        # 舒适性评估
        jerk = np.diff(traj[:, 2]) / self.dt
        cost_comfort = np.max(np.abs(jerk))
        # 效率评估
        v = traj[:, 1]
        cost_efficiency = np.max(np.abs((v - self.target_speed)) / self.target_speed)
        # 总效用
        k_sf, k_sp, k_eff, k_com = 2, 2, 0.5, 0.5

        def cost(rho_):
            return ((1 - rho_) * k_sf * cost_safe +
                    rho_ * k_sp * cost_space +
                    (1 - np.abs(1 - 2 * rho_)) * (k_com * cost_comfort + k_eff * cost_efficiency))

        if return_lambda:
            return cost
        return cost(rho)

    def get_strategies(self, is_lc: bool = False):
        if is_lc:
            # 换道方向
            strategy = [-1, 1]
        else:
            # 期望时距
            strategy = list(np.arange(0.5, 5, 0.5))
        return strategy

    def pred_self_traj(self, stra, T, is_lc: bool = False, uniform_speed=False):
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

    def lane_change_judge(self):
        pass
