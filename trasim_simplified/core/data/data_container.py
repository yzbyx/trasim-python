# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:20
# @Author : yzbyx
# @File : data_container.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from trasim_simplified.core.frame.frame_abstract import FrameAbstract


class DataContainer:
    def __init__(self, frame_abstract: 'FrameAbstract'):
        self.frame = frame_abstract
        self.data_pd: Optional[pd.DataFrame] = None
        self.save_info: set = set()

    def config(self, save_info=None, add_all=True):
        """默认包含车辆ID"""
        if add_all:
            self.save_info.update(Info.get_all_info().values())
            return
        if save_info is None:
            save_info = {}
        self.save_info.update(save_info)

        if Info.id in self.save_info: self.save_info.remove(Info.id)
        if Info.step in self.save_info: self.save_info.remove(Info.step)
        if Info.time in self.save_info: self.save_info.remove(Info.time)

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
        assert len(self.save_info) > 0, "若要调用record，save_info不能为空"

        temp = self.frame.car_id
        car_num_on_lane = temp.shape[1]
        temp = np.concatenate([temp, np.array([[self.frame.step_] * car_num_on_lane])], axis=0)
        temp = np.concatenate([temp, np.array([[self.frame.time_] * car_num_on_lane])], axis=0)
        for info in self.save_info:
            if Info.a == info:
                temp = np.concatenate([temp, self.frame.car_acc], axis=0)
            if Info.v == info:
                temp = np.concatenate([temp, self.frame.car_speed], axis=0)
            if Info.x == info:
                temp = np.concatenate([temp, self.frame.car_pos], axis=0)

            # TODO：将以下代码转移至processor以提高运行效率
            current_dhw = self._get_dhw(self.frame.car_pos)
            if Info.dhw == info:
                temp = np.concatenate([temp, current_dhw], axis=0)
            if Info.gap == info:
                temp = np.concatenate([temp, current_dhw - self.frame.car_length], axis=0)
            if Info.thw == info:
                temp = np.concatenate([temp, current_dhw / (self.frame.car_speed + np.finfo(np.float32).eps)], axis=0)
            if Info.dv == info:
                temp = np.concatenate([temp, self._get_dv(self.frame.car_speed)], axis=0)
        df = pd.DataFrame(data=temp.T, columns=self.save_info)
        self.data_pd.append(df)

class Info:
    id = "ID"
    """车辆ID"""
    step = "Step"
    """仿真步次"""
    time = "Time [s]"
    """仿真时间"""
    a = "Acceleration [m/s^2]"
    """加速度"""
    v = "Velocity [m/s]"
    """速度"""
    x = "Position [m]"
    """位置"""
    dhw = "Distance Headway [m]"
    """车头间距"""
    thw = "Time Headway [s]"
    """车头时距"""
    gap = "Gap [m]"
    """净间距"""
    dv = "Dv [m/s]"
    """前车与后车速度差"""

    @classmethod
    def get_all_info(cls):
        dict_ = Info.__dict__
        values = {}
        for key in dict_.keys():
            if isinstance(dict_[key], str) and key[:2] != "__":
                values.update({key: dict_[key]})
        return values


if __name__ == '__main__':
    print(Info.get_all_info())
