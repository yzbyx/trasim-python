# -*- coding = uft-8 -*-
# @time : 2022/1/11
# @Author : yzbyx
# @File : vehicle.py
# @Software : PyCharm
import queue
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.agent import utils
from trasim_simplified.core.agent.traj_predictor import TrajPred, get_pred_net
from trasim_simplified.core.constant import COLOR, V_TYPE, TrackInfo as C_Info, VehSurr, TrajPoint, RouteType
from trasim_simplified.core.kinematics.cfm import get_cf_model, CFModel, get_cf_id
from trasim_simplified.core.kinematics.lcm import get_lc_model, LCModel, get_lc_id
from trasim_simplified.core.agent.obstacle import Obstacle
from trasim_simplified.msg.trasimError import TrasimError

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class Vehicle(Obstacle):
    def __init__(self, lane: Optional['LaneAbstract'], type_: V_TYPE, id_: int, length: float):
        super().__init__(type_)
        self.ID = id_
        self.length = length
        self.lane = lane
        self.skip = False

        self.leader: Optional[Vehicle] = None
        self.follower: Optional[Vehicle] = None

        self.lane_id_list = []
        self.x_list = []
        self.y_list = []
        self.x_center_global_list = []
        self.y_center_global_list = []
        self.speed_list = []
        self.acc_list = []
        self.yaw_list = []
        self.delta_list = []

        self.step_list = []
        self.time_list = []
        self.dv_list = []
        """前车与后车速度差"""
        self.gap_list = []
        self.thw_list = []
        self.dhw_list = []
        self.ttc_list = []
        self.tit_list = []
        self.ttc_star = 1.5
        self.tet_list = []
        self.picud_list = []
        self.picud_KK_list = []

        self.preceding_id_list = []

        self.cf_model: Optional[CFModel] = None
        self.lc_model: Optional[LCModel] = None

        self.cf_acc = 0
        self.lc_result = {"lc": 0, "a": None, "v": None, "x": None}
        """换道模型结果，lc（-1（向左换道）、0（保持当前车道）、1（向右换道）），a（换道位置调整加速度），v（速度），x（位置）"""
        self.lc_res_pre = self.lc_result.copy()
        self.is_run_out = False
        """是否驶离路外"""
        self.pre_left_leader_follower: Optional[tuple[Vehicle, Vehicle]] = None
        self.pre_right_leader_follower: Optional[tuple[Vehicle, Vehicle]] = None

        self.next_acc = 0
        self.next_delta = 0
        self.target_lane = lane

        self.lf: Optional['Vehicle'] = None
        self.lr: Optional['Vehicle'] = None
        self.f: Optional['Vehicle'] = None
        self.r: Optional['Vehicle'] = None
        self.rf: Optional['Vehicle'] = None
        self.rr: Optional['Vehicle'] = None
        self.left_lane: Optional['LaneAbstract'] = None
        self.right_lane: Optional['LaneAbstract'] = None

        self.destination_lane_indexes = None
        self.route_type: Optional[RouteType] = None

        self.no_lc = False  # 是否禁止换道
        self.lc_direction = 0  # 换道方向，-1为左，1为右
        self.lane_changing = False  # 是否处于换道状态
        self.is_gaming = False  # 是否处于博弈状态
        self.game_time_wanted = None  # 博弈策略（期望时距）

        self.hist_traj: list[TrajPoint] = []
        """包含当前时间步的轨迹点"""

        self.PREVIEW_TIME = 1
        self.MIN_PREVIEW_S = 2
        self.TAU_HEADING = 0.2  # [s]
        self.TAU_LATERAL = 0.6  # [s]
        self.TAU_PURSUIT = 0.5 * self.TAU_HEADING  # [s]
        self.KP_HEADING = 1 / self.TAU_HEADING
        self.KP_LATERAL = 1 / self.TAU_LATERAL  # [1/s]

        self.pred_net: Optional[TrajPred] = get_pred_net()
        self.pred_traj: Optional[list[TrajPoint]] = None

        self.lc_ttc_risk = None
        self.lc_acc_benefit = None
        self.lc_route_desire = None

        self.risk_2d = None

    @property
    def time_wanted(self):
        return self.cf_model.get_time_wanted()

    @property
    def time_safe(self):
        return self.cf_model.get_time_safe()

    @property
    def acc_desire(self):
        return self.cf_model.get_expect_acc()

    @property
    def vel_desire(self):
        return self.cf_model.get_expect_speed()

    @property
    def dec_desire(self):
        return self.cf_model.get_expect_dec()

    @property
    def max_speed(self):
        return self.cf_model.get_max_speed()

    @property
    def dec_max(self):
        max_dec = self.cf_model.get_max_dec()
        return abs(max_dec)

    @property
    def acc_max(self):
        return self.cf_model.get_max_acc()

    @property
    def safe_s0(self):
        return self.cf_model.get_safe_s0()

    def update_necessary_info(self):
        self.f = self.leader
        self.r = self.follower
        self.lr, self.lf, self.rr, self.rf = self.lane.road.get_neighbour_vehicles(self, "all")
        self.left_lane, self.right_lane = (
            self.lane.road.get_available_adjacent_lane(self.lane, self.x)
        )

        if len(self.hist_traj) in [0, 1]:
            hist_traj = []
            current_traj_point = self.get_traj_point()
            hist_traj.append(current_traj_point.copy())
            # 按照恒定速度和航向角预测
            for i in range(19):
                current_traj_point.x -= current_traj_point.vx * self.dt
                current_traj_point.y -= current_traj_point.vy * self.dt
                hist_traj.append(current_traj_point.copy())
            self.hist_traj = hist_traj[::-1]

    def cal_traj_pred(self):
        self.pred_traj = None
        self.pred_traj = self.pred_net.pred_traj(self.pack_veh_surr(), time_len=10)

    def pack_veh_surr(self):
        """打包车辆周围车辆信息"""
        return VehSurr(
            ev=self,
            cp=self.f,
            cr=self.r,
            lp=self.lf,
            lr=self.lr,
            rp=self.rf,
            rr=self.rr,
        )

    @property
    def last_step_lc_statu(self):
        """0为保持车道，-1为向左换道，1为向右换道"""
        return self.lc_res_pre.get("lc", 0)

    def set_cf_model(self, cf_name: str, cf_param: dict):
        self.cf_model = get_cf_model(cf_name)(cf_param)

    def set_lc_model(self, lc_name: str, lc_param: dict):
        self.lc_model = get_lc_model(lc_name)(lc_param)

    def cal_vehicle_control(self):
        """3、车辆运动控制"""
        self.next_acc = self.cf_model.step(self.pack_veh_surr())
        self.next_delta = 0

    def lc_intention_judge(self):
        """1、判断是否换道并计算换道轨迹"""
        if self.no_lc:
            return
        self.target_lane, _ = self.lc_model.step(self.pack_veh_surr())

    def lc_decision_making(self):
        """2、换道决策"""
        pass

    def clone(self):
        """克隆车辆"""
        new_vehicle = Vehicle(lane=self.lane, type_=self.type, id_=self.ID, length=self.length)
        new_vehicle.x = self.x
        new_vehicle.y = self.y
        new_vehicle.speed = self.speed
        new_vehicle.acc = self.acc
        new_vehicle.yaw = self.yaw
        new_vehicle.delta = self.delta
        new_vehicle.cf_model = self.cf_model
        new_vehicle.lc_model = self.lc_model

        return new_vehicle

    def pred_self_traj(self, time_len, stra=None,
                       target_lane: "LaneAbstract" = None,
                       PC_traj=None, to_ndarray=True):
        """在策略下预测自车轨迹（包含初始状态）
        :param time_len: 预测时间长度
        :param stra: 期望时距
        :param target_lane: 目标车道
        :param PC_traj: 前车轨迹
        :param TF_traj: 目标车道前车轨迹
        :param to_ndarray: 是否转为ndarray
        """
        step_num = round(time_len / self.dt) + 1
        veh = self.clone()
        if target_lane is not None:
            veh.target_lane = target_lane
            veh.lane_changing = True
        if stra is not None:
            veh.game_time_wanted = stra
            veh.is_gaming = True
        leader = veh.clone()

        if self.f is None and PC_traj is None:
            traj = [self.get_traj_point().to_ndarray() if to_ndarray else self.get_traj_point()]
            # 估计轨迹
            leader.x = veh.x + 1e6
            leader.speed = veh.speed
            leader.acc = 0
            for step in range(step_num - 1):
                delta = veh.cf_lateral_control()
                acc = veh.cf_model.step(VehSurr(ev=veh, cp=leader))
                veh.update_state(acc, delta)
                traj.append(veh.get_traj_point().to_ndarray() if to_ndarray else veh.get_traj_point())
            PC_traj = None
            traj = np.array(traj) if to_ndarray else traj

        else:
            if PC_traj is None:
                PC_traj = self.f.pred_traj[:step_num]
            traj = [self.get_traj_point().to_ndarray() if to_ndarray else self.get_traj_point()]

            for step in range(round(step_num) - 1):
                info = PC_traj[step]
                leader.x = info.x
                leader.y = info.y
                leader.speed = info.speed
                leader.acc = info.acc

                delta = veh.cf_lateral_control()
                acc = veh.cf_model.step(VehSurr(ev=veh, cp=leader))
                veh.update_state(acc, delta)
                traj.append(veh.get_traj_point().to_ndarray() if to_ndarray else veh.get_traj_point())
            traj = np.array(traj) if to_ndarray else traj
            if to_ndarray:
                PC_traj = np.array([traj_point.to_ndarray() for traj_point in PC_traj])
        assert len(traj) == step_num
        return traj, PC_traj

    def cf_lateral_control(self):
        """自动驾驶跟驰过程/人类驾驶全过程的预瞄横向控制"""
        preview_s = max(self.MIN_PREVIEW_S, self.v * self.PREVIEW_TIME)
        l_lat = self.target_lane.y_center - self.y
        dist = np.sqrt(preview_s ** 2 + l_lat ** 2)  # 车头到预瞄点的距离
        theta = np.arctan2(l_lat, preview_s)

        R_h = (2 * self.l_rear_axle_2_head + dist) / (2 * (theta + self.lane.heading - self.yaw))
        R_r = np.sqrt(R_h ** 2 - self.l_rear_axle_2_head ** 2)

        delta = np.arctan2(self.wheelbase * np.sign(theta), R_r)

        d_delta = np.clip(delta - self.delta, -self.D_DELTA_MAX * self.dt, self.D_DELTA_MAX * self.dt)
        delta = self.delta + d_delta
        delta = np.clip(delta, -self.DELTA_MAX, self.DELTA_MAX)

        # lane_future_heading = 0
        #
        # # Lateral position control
        # lateral_speed_command = -self.KP_LATERAL * (self.y_c - self.target_lane.y_center)
        # # Lateral speed to heading
        # heading_command = np.arcsin(
        #     np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1)
        # )
        # heading_ref = lane_future_heading + np.clip(
        #     heading_command, -np.pi / 4, np.pi / 4
        # )
        # # Heading control
        # heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
        #     heading_ref - self.yaw
        # )
        # # Heading rate to steering angle
        # slip_angle = np.arcsin(
        #     np.clip(
        #         self.length / 2 / utils.not_zero(self.speed) * heading_rate_command,
        #         -1,
        #         1,
        #         )
        # )
        # delta = np.arctan(2 * np.tan(slip_angle))

        # d_delta = np.clip(delta - self.delta, -self.D_DELTA_MAX * self.dt, self.D_DELTA_MAX * self.dt)
        # delta = self.delta + d_delta
        delta = np.clip(delta, -self.DELTA_MAX, self.DELTA_MAX)

        return delta

    def get_history_trajectory(self, hist_time: float):
        """获取历史轨迹"""
        hist_time_step = int(hist_time / self.lane.dt)
        if len(self.hist_traj) == 0:
            traj = [None] * hist_time_step
        else:
            traj = self.hist_traj[-hist_time_step:]
        return traj

    def get_dist(self, pos: float):
        """获取pos与车头的距离，如果为环形边界，选取距离最近的表述，如果为开边界，pos-self.x"""
        if self.lane.is_circle:
            if pos > self.x:
                dist_head = pos - self.x
                dist_after = self.lane.lane_length - pos + self.x
                dist = dist_head if dist_head < dist_after else (- dist_after)
            else:
                dist_head = pos + self.lane.lane_length - self.x
                dist_after = self.x - pos
                dist = dist_head if dist_head < dist_after else (- dist_after)
        else:
            dist = pos - self.x
        return dist

    @property
    def is_first(self):
        """主要供TP模型使用"""
        return self.leader is None or self.leader.type == V_TYPE.OBSTACLE

    def get_data_list(self, info):
        if C_Info.lane_add_num == info:
            return self.lane_id_list
        elif C_Info.id == info:
            return [self.ID] * len(self.lane_id_list)
        elif C_Info.Preceding_ID == info:
            return self.preceding_id_list
        elif C_Info.car_type == info:
            return [self.type] * len(self.lane_id_list)
        elif C_Info.length == info:
            return [self.length] * len(self.lane_id_list)
        elif C_Info.width == info:
            return [self.width] * len(self.lane_id_list)
        elif C_Info.acc == info:
            return self.acc_list
        elif C_Info.speed == info:
            return self.speed_list
        elif C_Info.x == info:
            return self.x_list
        elif C_Info.Local_Y == info:
            return self.y_list
        elif C_Info.xCenterGlobal == info:
            return self.x_center_global_list
        elif C_Info.yCenterGlobal == info:
            return self.y_center_global_list
        elif C_Info.yaw == info:
            return self.yaw_list
        elif C_Info.delta == info:
            return self.delta_list

        elif C_Info.dv == info:
            return self.dv_list
        elif C_Info.gap == info:
            return self.gap_list
        elif C_Info.dhw == info:
            return self.dhw_list
        elif C_Info.thw == info:
            return self.thw_list
        elif C_Info.time == info:
            return self.time_list
        elif C_Info.step == info:
            return self.step_list
        elif C_Info.cf_id == info:
            return [get_cf_id(self.cf_model.name)] * len(self.lane_id_list)
        elif C_Info.lc_id == info:
            return [get_lc_id(None if self.lc_model is None else self.lc_model.name)] * len(self.lane_id_list)

        elif C_Info.safe_ttc == info:
            return self.ttc_list
        elif C_Info.safe_tit == info:
            return self.tit_list
        elif C_Info.safe_tet == info:
            return self.tet_list
        elif C_Info.safe_picud == info:
            return self.picud_list
        elif C_Info.safe_picud_KK == info:
            return self.picud_KK_list
        else:
            TrasimError(f"{info}未创建！")

    def record(self):
        for info in self.lane.data_container.save_info:
            if C_Info.lane_add_num == info:
                self.lane_id_list.append(self.lane.add_num)
            if C_Info.Preceding_ID == info:
                self.preceding_id_list.append(self.leader.ID if self.leader is not None else np.nan)
            if C_Info.acc == info:
                self.acc_list.append(self.acc)
            elif C_Info.speed == info:
                self.speed_list.append(self.speed)
            elif C_Info.Local_X == info:
                self.x_list.append(self.x)
            elif C_Info.Local_Y == info:
                self.y_list.append(self.y - self.lane.y_center)
            elif C_Info.xCenterGlobal == info:
                self.x_center_global_list.append(self.x_c)
            elif C_Info.yCenterGlobal == info:
                self.y_center_global_list.append(self.y_c)
            elif C_Info.yaw == info:
                self.yaw_list.append(self.yaw)
            elif C_Info.delta == info:
                self.delta_list.append(self.delta)

            elif C_Info.dv == info:
                self.dv_list.append(self.dv)
            elif C_Info.gap == info:
                self.gap_list.append(self.gap)
            elif C_Info.dhw == info:
                self.dhw_list.append(self.dhw)
            elif C_Info.thw == info:
                self.thw_list.append(self.thw)
            elif C_Info.time == info:
                self.time_list.append(self.lane.time_)
            elif C_Info.step == info:
                self.step_list.append(self.lane.step_)

            elif C_Info.safe_ttc == info:
                self.ttc_list.append(self.ttc)
            elif C_Info.safe_tit == info:
                self.tit_list.append(self.tit)
            elif C_Info.safe_tet == info:
                self.tet_list.append(self.tet)
            elif C_Info.safe_picud == info:
                self.picud_list.append(self.picud)
            elif C_Info.safe_picud_KK == info:
                self.picud_KK_list.append(self.picud_KK)
            else:
                TrasimError(f"{info}未创建！")

    @property
    def gap(self):
        if self.leader is not None:
            dhw = self.dhw
            gap = dhw - self.leader.length
            return gap
        else:
            return np.nan

    @property
    def dv(self):
        """前车与当前车速度差"""
        if self.leader is not None:
            return self.leader.v - self.v
        else:
            return np.nan

    @property
    def dhw(self):
        if self.leader is not None:
            dhw = self.leader.x - self.x
            if dhw < 0:
                if self.lane.is_circle and self.lane.car_list[-1].ID == self.ID:
                    dhw += self.lane.lane_length
                else:
                    raise TrasimError(f"车头间距小于0！\n" + self.get_basic_info())
            return dhw
        else:
            return np.nan

    @property
    def thw(self):
        if self.leader is not None:
            if self.dv != 0:
                return self.dhw / (- self.dv)
        return np.nan

    @property
    def ttc(self):
        if self.leader is not None:
            if self.dv != 0:
                return self.gap / (- self.dv)
        return np.nan

    @property
    def tit(self):
        """只是0 <= ttc <= ttc_star时计算单个ttc_star - ttc"""
        ttc = self.ttc
        if 0 <= ttc <= self.ttc_star:
            return self.ttc_star - ttc
        return 0

    @property
    def tet(self):
        ttc = self.ttc
        if 0 <= ttc <= self.ttc_star:
            return 1
        return 0

    @property
    def picud(self):
        if self.leader is not None:
            l_dec = self.leader.cf_model.get_expect_dec()
            l_v = self.leader.v
            l_x = self.leader.x if self.leader.x > self.x else (self.leader.x + self.lane.lane_length)
            l_length = self.leader.length
            dec = self.cf_model.get_expect_dec()
            xd_l = (l_v ** 2) / (2 * l_dec)
            xd = (self.v ** 2) / (2 * dec)
            return (l_x + xd_l) - (self.x + self.v * self.lane.dt + xd) - l_length
        else:
            return np.nan

    @property
    def picud_KK(self):
        if self.leader is not None:
            l_v = self.leader.v
            l_x = self.leader.x if self.leader.x > self.x else (self.leader.x + self.lane.lane_length)
            l_length = self.leader.length
            dec = self.cf_model.get_expect_dec()
            tau = self.lane.dt

            alpha = int(l_v / (dec * tau))  # 使用当前车的最大期望减速度
            beta = l_v / (dec * tau) - int(l_v / (dec * tau))
            xd_l = dec * tau * tau * (alpha * beta + 0.5 * alpha * (alpha - 1))

            alpha = int(self.v / (dec * tau))
            beta = self.v / (dec * tau) - int(self.v / (dec * tau))
            xd = dec * tau * tau * (alpha * beta + 0.5 * alpha * (alpha - 1))

            return (l_x + xd_l) - (self.x + self.v * tau + xd) - l_length
        else:
            return np.nan

    def has_data(self):
        return len(self.x_list) != 0

    def set_car_param(self, param: dict):
        self.color = param.get("color", COLOR.yellow)
        self.width = param.get("width", 1.8)

    def get_basic_info(self, sep="\n"):
        return f"step: {self.lane.step_}, time: {self.lane.time_}, lane_index: {self.lane.index}{sep}" \
               f"ego_lc: {self.last_step_lc_statu}, " \
               f"ego_id: {self.ID}, ego_type: {self.cf_model.name}," \
               f" ego_x: {self.x:.3f}, ego_v: {self.v:.3f}, ego_a: {self.a:.3f}{sep}" + \
               f"leader_lc: {self.leader.last_step_lc_statu}, leader_id: {self.leader.ID}," \
               f" leader_type: {self.leader.cf_model.name}," \
               f" leader_x: {self.leader.x:.3f}, leader_v: {self.leader.v:.3f}, leader_a: {self.leader.a:.3f}" \
            if self.leader is not None else ""

    def __repr__(self):
        print(self.risk_2d)
        return (f"id: {self.ID}, x: {self.x:.3f}, y: {self.y:.3f}, speed: {self.speed:.3f}"
                f" acc: {self.acc:.3f}, lane: {self.lane.index}, gap: {self.gap:.3f},"
                f" is_lc: {self.lane_changing}, ttc: {self.ttc:.3f}, ttc2d: {float(self.risk_2d):.3f}")
