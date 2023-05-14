# -*- coding = uft-8 -*-
# @Time : 2022/1/11
# @Author : yzbyx
# @File : vehicle.py
# @Software : PyCharm
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.constant import COLOR
from trasim_simplified.core.kinematics.cfm import get_cf_model, CFModel
from trasim_simplified.core.kinematics.lcm import get_lc_model, LCModel
from trasim_simplified.core.obstacle import Obstacle
from trasim_simplified.msg.trasimError import TrasimError
from trasim_simplified.msg.trasimWarning import TrasimWarning

if TYPE_CHECKING:
    from trasim_simplified.core.frame.lane_abstract import LaneAbstract


class Vehicle(Obstacle):
    def __init__(self, lane: 'LaneAbstract', type_: str, id_: int, length: float):
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

        self.cf_model: Optional[CFModel] = None
        self.lc_model: Optional[LCModel] = None

        self.cf_acc = 0
        self.lc_result = {"lc": 0, "a": 0, "v": None, "x": None}
        """换道模型结果，lc（-1（向左换道）、0（保持当前车道）、1（向右换道）），a（换道位置调整加速度），v（速度），x（位置）"""

    def set_cf_model(self, cf_name: str, cf_param: dict):
        self.cf_model = get_cf_model(self, cf_name, cf_param)

    def set_lc_model(self, lc_name: str, lc_param: dict):
        self.lc_model = get_lc_model(self, lc_name, lc_param)

    def step(self, index):
        self.cf_acc = self.cf_model.step(index)

    def step_lane_change(self, index: int, left_lane: 'LaneAbstract', right_lane: 'LaneAbstract'):
        self.lc_result = self.lc_model.step(index, left_lane, right_lane)

    def get_dist(self, pos: float):
        """获取pos与车头的距离，如果为环形边界，选取距离最近的表述"""
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

    def get_data_list(self, info):
        from trasim_simplified.core.data.data_container import Info as C_Info
        if C_Info.lane_id == info:
            return self.lane_id_list
        elif C_Info.id == info:
            return [self.ID] * len(self.pos_list)
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
        elif C_Info.safe_ttc == info:
            return self.ttc_list
        elif C_Info.safe_tit == info:
            return self.tit_list
        elif C_Info.safe_tet == info:
            return self.tet_list
        elif C_Info.safe_picud == info:
            return self.picud_list

    def record(self):
        from trasim_simplified.core.data.data_container import Info as C_Info
        for info in self.lane.data_container.save_info:
            if C_Info.lane_id == info:
                self.lane_id_list.append(self.lane.ID)
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

    @property
    def gap(self):
        if self.leader is not None:
            dhw = self.dhw
            return dhw - self.leader.length
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
                    # raise TrasimError("车头间距小于0！")
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

    def has_data(self):
        return len(self.pos_list) != 0

    def set_car_param(self, param: dict):
        self.color = param.get("color", COLOR.yellow)
        self.width = param.get("width", 1.8)
