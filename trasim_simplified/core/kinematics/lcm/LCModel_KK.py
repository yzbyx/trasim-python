# -*- coding: utf-8 -*-
# @Time : 2023/5/12 16:31
# @Author : yzbyx
# @File : LCModel_KK.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.kinematics.cfm.CFModel_KK import cal_G
from trasim_simplified.msg.trasimError import TrasimError

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract

from trasim_simplified.core.kinematics.lcm.LCModel import LCModel
from trasim_simplified.core.constant import LCM


class LCModel_KK(LCModel):
    def __init__(self, vehicle: 'Vehicle', l_param: dict[str, float]):
        super().__init__(vehicle)
        self.name = LCM.KK
        self.thesis = 'Physics of Automated-Driving Vehicular Traﬃc'

        self._a_0 = vehicle.cf_model.get_expect_acc()
        self._delta_1 = l_param.get("delta_1", 2 * self._a_0 * self.vehicle.lane.dt)
        self._delta_2 = l_param.get("delta_2", 5)
        self._gamma_ahead = l_param.get("gamma_ahead", 1)
        self._gamma_behind = l_param.get("gamma_behind", 0.5)
        self._k = l_param.get("k", 3.5)
        self._tau = l_param.get("tau", 1)
        """反应时间 (tau)"""
        self._L_a = l_param.get("L_a", 150)
        """前视距离 (m)"""
        self._p_0 = l_param.get("p_0", 0.45)
        """换道概率"""

        # -----on ramp----- #
        self._lambda_b = l_param.get("lambda_b", 0.75)
        self._delta_vr_1 = l_param.get("delta_vr_1", 10.)
        self.xm = np.nan

        self.left_lane: Optional['LaneAbstract'] = None
        self.right_lane: Optional['LaneAbstract'] = None
        self.lane: Optional['LaneAbstract'] = None

    def _update_dynamic(self):
        assert self.vehicle.lane.is_circle is False, "此换道模型在边界处由于xm"
        self.lane = self.vehicle.lane
        self.dt = self.lane.dt

    def step(self, index, *args):
        self._update_dynamic()
        self.left_lane, self.right_lane = args
        # type_ = self.lane.get_section_type(self.vehicle.x, self.vehicle.type)
        return self.base_cal()
        # if SECTION_TYPE.BASE in type_:
        #     return self.base_cal()
        # if SECTION_TYPE.ON_RAMP in type_:
        #     return self.on_ramp_cal()
        # raise TrasimError(f"没有对应{type_}的处理函数！")

    def base_cal(self):
        if self.vehicle.leader is None:
            return {"lc": 0}

        left_d_l = -1
        right_d_l = -1
        left_ = False
        right_ = False

        l_v = self.vehicle.leader.v if self.vehicle.dhw < self._L_a else np.inf
        # 判断是否选择左转
        if self.left_lane is not None:
            _f, _l = self.left_lane.get_relative_car(self.vehicle)
            safe_, left_d_l = self._safe_check(_f, _l)
            if not safe_:
                left_ = False
            else:
                if _l is not None:
                    if _l.v >= l_v + self._delta_1 and self.vehicle.v > l_v:
                        left_ = True
                else:
                    left_ = True
        # 判断是否选择右转
        if self.right_lane is not None:
            _f, _l = self.right_lane.get_relative_car(self.vehicle)
            safe_, right_d_l = self._safe_check(_f, _l)
            if not safe_:
                right_ = False
            else:
                if _l is not None:
                    if _l.v >= l_v + self._delta_2 or _l.v >= self.vehicle.v + self._delta_2:
                        right_ = True
                else:
                    right_ = True
        # 概率选择是否换道
        if left_ or right_:
            if self.random.random() < self._p_0:
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
            D_ahead = cal_G(self._k, self._tau, self._a_0, self.vehicle.v, _l.v)
            head_safe = (d_l > min(self._gamma_ahead * self.vehicle.v * self._tau + _l.length, D_ahead))
        else:
            head_safe = True
            d_l = np.inf
        if _f is not None:
            d_f = _f.get_dist(self.vehicle.x) - self.vehicle.length
            D_behind = cal_G(self._k, self._tau, self._a_0, _f.v, self.vehicle.v)
            behind_safe = (d_f > min(self._gamma_behind * _f.v * self._tau + self.vehicle.length, D_behind))
        else:
            behind_safe = True
        return head_safe and behind_safe, d_l

    def on_ramp_cal(self):
        """限制仅向左换道"""
        if self.left_lane is not None:
            _f, _l = self.left_lane.get_relative_car(self.vehicle)
            safe_, left_d_l, v_hat, x = self._safe_check_on_ramp(_f, _l)
        else:
            safe_ = False
            v_hat = x = None
        if safe_:
            return {"lc": -1, "x": x, "v": v_hat}
        return {"lc": 0}

    def _safe_check_on_ramp(self, _f: 'Vehicle', _l: 'Vehicle'):
        x = self.vehicle.x
        if _l is not None:
            D_ahead, head_safe, d_l, v_hat = self._safe_func_on_ramp_common(self.vehicle, _l)
        else:
            v_hat = self.vehicle.v + self._delta_vr_1
            head_safe = True
            d_l = np.inf
        if _f is not None:
            D_behind, behind_safe, _, _ = self._safe_func_on_ramp_common(_f, self.vehicle, v_hat)
        else:
            behind_safe = True

        xm = np.nan
        if _l is not None and _f is not None:
            xm = _f.x + _f.dhw / 2
        if not (head_safe and behind_safe):
            if _l is not None and _f is not None:
                if _f.gap > self._lambda_b * _f.v + self.vehicle.length:
                    condition_1 = (self.vehicle.pos_list[-1] < self.xm and self.vehicle.x >= xm)
                    condition_2 = (self.vehicle.pos_list[-1] >= self.xm and self.vehicle.x < xm)
                    if condition_1 or condition_2:
                        head_safe = behind_safe = True
                        x = xm
        self.xm = xm

        return head_safe and behind_safe, d_l, v_hat, x

    def _safe_func_on_ramp_common(self, follower: 'Vehicle', leader: 'Vehicle', v_hat=None):
        """

        :param follower:
        :param leader:
        :param v_hat: 是否为ego的后车和ego, 若是则需要传ego的v_hat
        :return:
        """
        d_l = - leader.get_dist(follower.x) - leader.length
        if v_hat is None:
            v_hat = min(leader.v, follower.v + self._delta_vr_1)
            D = cal_G(self._k, self._tau, self._a_0, v_hat, leader.v)
        else:
            D = cal_G(self._k, self._tau, self._a_0, follower.v, v_hat)
        safe = (d_l >= min(v_hat * self._tau, D))
        return D, safe, d_l, v_hat
