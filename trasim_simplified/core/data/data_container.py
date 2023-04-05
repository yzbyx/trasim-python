# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:20
# @Author : yzbyx
# @File : data_container.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from trasim_simplified.core.frame.frame_abstract import FrameAbstract


class DataContainer:
    def __init__(self, frame_abstract: 'FrameAbstract'):
        self.frame = frame_abstract
        self.acc_data: Optional[np.ndarray] = None
        self.speed_data: Optional[np.ndarray] = None
        self.pos_data: Optional[np.ndarray] = None
        self.gap_data: Optional[np.ndarray] = None
        self.dhw_data: Optional[np.ndarray] = None
        self.thw_data: Optional[np.ndarray] = None
        self.dv_data: Optional[np.ndarray] = None

    def _get_dhw(self, car_pos):
        car_pos = np.concatenate([car_pos, [[car_pos[0, 0]]]], axis=1)
        dhw = np.diff(car_pos)
        dhw[np.where(dhw < 0)] += self.frame.lane_length
        return dhw

    @staticmethod
    def _get_dv(car_speed):
        car_speed = np.concatenate([car_speed, [[car_speed[0, 0]]]], axis=1)
        return - np.diff(car_speed)

    def record(self):
        self.acc_data = np.concatenate([self.acc_data, self.frame.car_acc], axis=0) \
            if self.acc_data is not None else self.frame.car_acc.copy()
        self.speed_data = np.concatenate([self.speed_data, self.frame.car_speed], axis=0) \
            if self.speed_data is not None else self.frame.car_speed.copy()
        self.pos_data = np.concatenate([self.pos_data, self.frame.car_pos], axis=0) \
            if self.pos_data is not None else self.frame.car_pos.copy()

        self.extend_record()

    def extend_record(self):
        current_dhw = self._get_dhw(self.frame.car_pos)
        self.dhw_data = np.concatenate([self.dhw_data, current_dhw], axis=0) \
            if self.dhw_data is not None else current_dhw
        self.gap_data = np.concatenate([self.gap_data, current_dhw - self.frame.car_length], axis=0) \
            if self.gap_data is not None else current_dhw - self.frame.car_length
        self.thw_data = np.concatenate([self.thw_data, current_dhw / (self.frame.car_speed + np.finfo(np.float32).eps)], axis=0) \
            if self.thw_data is not None else current_dhw / self.frame.car_speed
        self.dv_data = np.concatenate([self.dv_data, self._get_dv(self.frame.car_speed)], axis=0) \
            if self.dv_data is not None else self._get_dv(self.frame.car_speed)
