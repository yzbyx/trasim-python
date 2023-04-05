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

        self.save_info: set = set()

        self.current_dhw = None

    def config(self, save_info=None, add_all=True):
        if add_all:
            self.save_info.update(Info.get_all_info())
            return
        if save_info is None:
            save_info = {}
        self.save_info.update(save_info)

    def _get_dhw(self, car_pos):
        if self.dhw_data is None:
            car_pos = np.concatenate([car_pos, [[car_pos[0, 0]]]], axis=1)
            dhw = np.diff(car_pos)
            dhw[np.where(dhw < 0)] += self.frame.lane_length
            self.dhw_data = dhw
        return self.dhw_data

    @staticmethod
    def _get_dv(car_speed):
        car_speed = np.concatenate([car_speed, [[car_speed[0, 0]]]], axis=1)
        return - np.diff(car_speed)

    def record(self):
        if Info.a in self.save_info: self.acc_data = np.concatenate([self.acc_data, self.frame.car_acc], axis=0) \
            if self.acc_data is not None else self.frame.car_acc.copy()
        if Info.v in self.save_info: self.speed_data = np.concatenate([self.speed_data, self.frame.car_speed], axis=0) \
            if self.speed_data is not None else self.frame.car_speed.copy()
        if Info.x in self.save_info: self.pos_data = np.concatenate([self.pos_data, self.frame.car_pos], axis=0) \
            if self.pos_data is not None else self.frame.car_pos.copy()

        if Info.dhw in self.save_info:
            current_dhw = self._get_dhw(self.frame.car_pos)
            self.dhw_data = np.concatenate([self.dhw_data, current_dhw], axis=0) \
                if self.dhw_data is not None else current_dhw
        if Info.gap in self.save_info and Info.dhw in self.save_info:
            current_dhw = self._get_dhw(self.frame.car_pos)
            self.gap_data = np.concatenate([self.gap_data, current_dhw - self.frame.car_length], axis=0) \
                if self.gap_data is not None else current_dhw - self.frame.car_length
        if Info.thw in self.save_info and Info.dhw in self.save_info:
            current_dhw = self._get_dhw(self.frame.car_pos)
            self.thw_data = np.concatenate([self.thw_data, current_dhw / (self.frame.car_speed + np.finfo(np.float32).eps)], axis=0) \
                if self.thw_data is not None else current_dhw / self.frame.car_speed
        if Info.dv in self.save_info:
            self.dv_data = np.concatenate([self.dv_data, self._get_dv(self.frame.car_speed)], axis=0) \
                if self.dv_data is not None else self._get_dv(self.frame.car_speed)
        # 必须加
        self.dhw_data = None

class Info:
    a = 0
    """加速度"""
    v = 1
    """速度"""
    x = 2
    """位置"""
    dhw = 3
    """车头间距"""
    thw = 4
    """车头时距"""
    gap = 5
    """净间距"""
    dv = 6
    """前车与后车速度差"""

    @classmethod
    def get_all_info(cls):
        dict_ = Info.__dict__
        values = []
        for key in dict_.keys():
            if isinstance(dict_[key], int):
                values.append(dict_[key])
        return values


if __name__ == '__main__':
    print(Info.get_all_info())
