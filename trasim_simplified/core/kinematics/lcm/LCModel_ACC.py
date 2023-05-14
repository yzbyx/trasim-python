# -*- coding: utf-8 -*-
# @Time : 2023/5/13 16:47
# @Author : yzbyx
# @File : LCModel_ACC.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.kinematics.cfm.CFModel_KK import cal_G
from trasim_simplified.msg.trasimError import TrasimError

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle
    from trasim_simplified.core.frame.lane_abstract import LaneAbstract

from trasim_simplified.core.kinematics.lcm.LCModel import LCModel
from trasim_simplified.core.constant import LCM, SECTION_TYPE


class LCModel_ACC(LCModel):
    def __init__(self, vehicle: Optional['Vehicle'], l_param: dict[str, float]):
        super().__init__(vehicle)
        self.name = LCM.ACC
        self.thesis = "Physics of Automated-Driving Vehicular Traﬃc"

        self._a_0 = vehicle.cf_model.get_expect_acc()
        self._delta_1 = l_param.get("delta_1", 2 * self._a_0 * self.vehicle.lane.dt)
        self._delta_2 = l_param.get("delta_2", 5)
        self._tau_ahead = l_param.get("tau_ahead", 0.2)
        self._tau_behind = l_param.get("tau_behind", 0.6)

        self._L_a = l_param.get("L_a", 150)
        """前视距离 (m)"""

        self.left_lane: Optional['LaneAbstract'] = None
        self.right_lane: Optional['LaneAbstract'] = None
        self.lane: Optional['LaneAbstract'] = None

    def _update_dynamic(self):
        self.lane = self.vehicle.lane

    def step(self, index, *args):
        self._update_dynamic()
        self.left_lane, self.right_lane = args
        type_ = self.lane.get_section_type(self.vehicle.x)
        if type_ == SECTION_TYPE.BASE:
            return self.base_cal()

    def base_cal(self):
        if self.vehicle.leader is None:
            return {"lc": 0}

        left_d_l = -1
        right_d_l = -1
        left_ = False
        right_ = False

        l_v = self.vehicle.leader.v if self.vehicle.dhw < self._L_a else np.Inf
        if self.left_lane is not None:
            _f, _l = self.left_lane.get_relative_car(self.vehicle.x)
            safe_, left_d_l = self._safe_check(_f, _l)
            if not safe_:
                left_ = False
            else:
                if _l.v >= l_v + self._delta_1 and self.vehicle.v > l_v:
                    left_ = True
        if self.right_lane is not None:
            # TODO: 车辆的x需要换算到对应车道的位置
            _f, _l = self.right_lane.get_relative_car(self.vehicle.x)
            safe_, right_d_l = self._safe_check(_f, _l)
            if not safe_:
                right_ = False
            else:
                if _l.v >= l_v + self._delta_2 or _l.v >= self.vehicle.v + self._delta_2:
                    right_ = True

        if left_ or right_:
            if left_ and right_:
                direct = -1 if left_d_l > right_d_l else 1
            elif left_:
                direct = -1
            elif right_:
                direct = 1
            else:
                raise TrasimError("出错")
            return {"lc": direct}

        return {"lc": 0}

    def _safe_check(self, _f: 'Vehicle', _l: 'Vehicle'):
        if _l is not None:
            d_l = - _l.get_dist(self.vehicle.x) - _l.length
            head_safe = (d_l > self.vehicle.v * self._tau_ahead)
        else:
            head_safe = True
            d_l = np.Inf
        if _f is not None:
            d_f = _f.get_dist(self.vehicle.x) - self.vehicle.length
            behind_safe = (d_f > _f.v * self._tau_behind)
        else:
            behind_safe = True
        return head_safe and behind_safe, d_l
