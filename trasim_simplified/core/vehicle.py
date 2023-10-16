# -*- coding = uft-8 -*-
# @Time : 2022/1/11
# @Author : yzbyx
# @File : vehicle.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.constant import COLOR, V_TYPE, TrackInfo as C_Info
from trasim_simplified.core.kinematics.cfm import get_cf_model, CFModel, get_cf_id
from trasim_simplified.core.kinematics.lcm import get_lc_model, LCModel, get_lc_id
from trasim_simplified.core.obstacle import Obstacle
from trasim_simplified.msg.trasimError import TrasimError

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class Vehicle(Obstacle):
    def __init__(self, lane: 'LaneAbstract', type_: int, id_: int, length: float):
        super().__init__(type_)
        self.ID = id_
        self.length = length
        self.lane = lane

        self.leader: Optional[Vehicle] = None
        self.follower: Optional[Vehicle] = None

        self.lane_id_list = []
        self.pos_list = []
        self.speed_list = []
        self.acc_list = []
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

    @property
    def last_step_lc_statu(self):
        """0为保持车道，-1为向左换道，1为向右换道"""
        return self.lc_res_pre.get("lc", 0)

    def set_cf_model(self, cf_name: str, cf_param: dict):
        self.cf_model = get_cf_model(self, cf_name, cf_param)

    def set_lc_model(self, lc_name: str, lc_param: dict):
        self.lc_model = get_lc_model(self, lc_name, lc_param)

    def step(self, index):
        self.cf_acc = self.cf_model.step(index)

    def step_lane_change(self, index: int, left_lane: 'LaneAbstract', right_lane: 'LaneAbstract'):
        self.lc_result = self.lc_model.step(index, left_lane, right_lane)
        self.lc_res_pre = self.lc_result

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
            return [self.ID] * len(self.pos_list)
        if C_Info.Preceding_ID == info:
            return self.preceding_id_list
        elif C_Info.car_type == info:
            return [self.type] * len(self.pos_list)
        elif C_Info.v_Length == info:
            return [self.length] * len(self.pos_list)
        elif C_Info.a == info:
            return self.acc_list
        elif C_Info.v == info:
            return self.speed_list
        elif C_Info.x == info:
            return self.pos_list
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
            return [get_cf_id(self.cf_model.name)] * len(self.pos_list)
        elif C_Info.lc_id == info:
            return [get_lc_id(None if self.lc_model is None else self.lc_model.name)] * len(self.pos_list)

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
                self.preceding_id_list.append(self.leader.ID if self.leader is not None else np.NaN)
            if C_Info.a == info:
                self.acc_list.append(self.a)
            elif C_Info.v == info:
                self.speed_list.append(self.v)
            elif C_Info.x == info:
                self.pos_list.append(self.x)
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
            return np.NaN

    @property
    def dv(self):
        """前车与当前车速度差"""
        if self.leader is not None:
            return self.leader.v - self.v
        else:
            return np.NaN

    @property
    def dhw(self):
        if self.leader is not None:
            dhw = self.leader.x - self.x
            if dhw < 0:
                if self.lane.is_circle and self.lane.car_list[-1].ID == self.ID:
                    dhw += self.lane.lane_length
                else:
                    raise TrasimError(f"车头间距小于0！\n" + self.get_basic_info())
                    pass
            return dhw
        else:
            return np.NaN

    @property
    def thw(self):
        if self.leader is not None:
            if self.dv != 0:
                return self.dhw / (- self.dv)
        return np.NaN

    @property
    def ttc(self):
        if self.leader is not None:
            if self.dv != 0:
                return self.gap / (- self.dv)
        return np.NaN

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
            return np.NaN

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
            return np.NaN

    def has_data(self):
        return len(self.pos_list) != 0

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
        return f"type: {self.cf_model.name}, step: {self.lane.step_}" +\
            f" x: {self.x:.3f}, v: {self.v:.3f}, a: {self.a:.3f}, gap: {self.gap}"
