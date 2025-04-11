# -*- coding: utf-8 -*-
# @Time : 2025/3/24 0:48
# @Author : yzbyx
# @File : LCModel_Mobil.py
# Software: PyCharm
from typing import TYPE_CHECKING

import numpy as np

from trasim_simplified.core.constant import LCM, VehSurr
from trasim_simplified.core.kinematics.lcm import LCModel

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle


class LCModel_Mobil(LCModel):
    def __init__(self, l_param: dict[str, float]):
        super().__init__()
        self.name = LCM.MOBIL
        self.thesis = 'Physics of Automated-Driving Vehicular Traffic'

        self.right_lane = None
        self.left_lane = None

        # Lateral policy parameters
        # in [0, 1]
        self.POLITENESS = l_param.get("POLITENESS", 0.5)  # [0, 1]  礼让程度
        self.LANE_CHANGE_MIN_ACC_GAIN = l_param.get("LANE_CHANGE_MIN_ACC_GAIN", 0)  # [m/s2]  换道加速增益
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = l_param.get("LANE_CHANGE_MAX_BRAKING_IMPOSED", 9)  # [m/s2]  换道最大制动
        self.LANE_CHANGE_DELAY = l_param.get("LANE_CHANGE_DELAY", 0.5)  # [s]  换道决策延迟

    def _update_dynamic(self):
        self.lane = self.veh_surr.ev.lane
        self.dt = self.lane.dt
        self.left_lane = self.veh_surr.ev.left_lane
        self.right_lane = self.veh_surr.ev.right_lane

    def step(self, veh_surr: "VehSurr"):
        self.veh_surr = veh_surr
        self._update_dynamic()
        res = self.base_cal()
        return res

    def base_cal(self):
        if self.left_lane is not None:
            l_ok, acc_gain_l = self.mobil(-1, return_acc_gain=True)
        else:
            l_ok, acc_gain_l = False, -np.inf
        if self.right_lane is not None:
            r_ok, acc_gain_r = self.mobil(1, return_acc_gain=True)
        else:
            r_ok, acc_gain_r = False, -np.inf

        lc_acc_gain = [acc_gain_l, acc_gain_r]
        # assert acc_gain_l is not None and acc_gain_r is not None

        target_lane = self.veh_surr.ev.lane
        if l_ok and r_ok:
            if acc_gain_l >= acc_gain_r:
                lc_direction = self.left_lane
            else:
                lc_direction = self.right_lane
        elif l_ok:
            lc_direction = self.left_lane
        elif r_ok:
            lc_direction = self.right_lane
        else:
            lc_direction = target_lane

        return lc_direction, {
            "acc_gain": lc_acc_gain,
        }

    def mobil(self, direction: int, return_acc_gain=False) -> bool | tuple:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :param return_acc_gain: whether to return the jerk value
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        if direction == -1:
            new_preceding, new_following = self.veh_surr.lp, self.veh_surr.lr
        else:
            new_preceding, new_following = self.veh_surr.rp, self.veh_surr.rr

        if new_preceding is None:
            new_preceding = self.veh_surr.ev.clone()
            new_preceding.x += 1e5
        if new_following is None:
            new_following = self.veh_surr.ev.clone()
            new_following.x -= 1e5

        if abs(new_preceding.x - self.veh_surr.ev.x) < new_preceding.length:
            # The new preceding vehicle is too close
            if return_acc_gain:
                return False, -np.inf
            return False

        if abs(new_following.x - self.veh_surr.ev.x) < new_following.length:
            # The new following vehicle is too close
            if return_acc_gain:
                return False, -np.inf
            return False

        new_following_a = self.veh_surr.ev.cf_model.step(
            VehSurr(ev=new_following, cp=new_preceding)
        )
        new_following_pred_a = self.veh_surr.ev.cf_model.step(
            VehSurr(ev=new_following, cp=self.veh_surr.ev),
        )

        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            if return_acc_gain:
                return False, -np.inf
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.veh_surr.cp, self.veh_surr.cr

        if old_preceding is None:
            old_preceding = self.veh_surr.ev.clone()
            old_preceding.x += 1e5
        if old_following is None:
            old_following = self.veh_surr.ev.clone()
            old_following.x -= 1e5

        self_pred_a = self.veh_surr.ev.cf_model.step(
            VehSurr(ev=self.veh_surr.ev, cp=new_preceding)
        )
        # Unsafe braking required
        if self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            if return_acc_gain:
                return False, -np.inf
            return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        self_a = self.veh_surr.ev.cf_model.step(
            VehSurr(ev=self.veh_surr.ev, cp=old_preceding)
        )
        old_following_a = self.veh_surr.ev.cf_model.step(
            VehSurr(ev=old_following, cp=self.veh_surr.ev)
        )
        old_following_pred_a = self.veh_surr.ev.cf_model.step(
            VehSurr(ev=old_following, cp=old_preceding)
        )
        jerk = (
                self_pred_a
                - self_a
                + self.POLITENESS
                * (
                        new_following_pred_a
                        - new_following_a
                        + old_following_pred_a
                        - old_following_a
                )
        )
        if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
            if return_acc_gain:
                return False, jerk
            else:
                return False

        # All clear, let's go!
        if return_acc_gain:
            return True, jerk
        else:
            return True

    def on_ramp_cal(self):
        pass
