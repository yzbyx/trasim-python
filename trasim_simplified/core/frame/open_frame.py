# -*- coding = uft-8 -*-
# @Time : 2023-04-14 12:07
# @Author : yzbyx
# @File : open_frame.py
# @Software : PyCharm
from typing import Iterable, Optional

import numpy as np

from trasim_simplified.core.frame.frame_abstract import FrameAbstract
from trasim_simplified.msg.trasimWarning import TrasimWarning


class FrameOpen(FrameAbstract):
    def car_loader(self, flow_rate: int | float, time_table: Optional[Iterable] = None):
        pass

    def step(self):
        if self.car_pos.shape[1] > 1:
            leader_x = np.concatenate([self.car_pos[:, 1:], [[self.car_pos[0][-1] - self.car_pos[0][-2]]]], axis=1)
            leader_speed = np.concatenate([self.car_speed[:, 1:], [[self.car_speed[0][-1]]]], axis=1)
        else:
            leader_x = np.array([[self.lane_length]])
            leader_speed = self.car_speed
        self.car_acc = self.cf_model.step(
            self.car_speed,
            self.car_pos,
            leader_speed,
            leader_x,
            self.car_length
        )

    def update_state(self):
        car_speed_before = self.car_speed.copy()
        self.car_speed += self.car_acc * self.dt

        speed_neg_pos = np.where(self.car_speed < 0)
        if len(speed_neg_pos[0]) != 0:
            TrasimWarning("存在速度为负的车辆！")
            self.car_speed[speed_neg_pos] = 0

        self.car_pos += (car_speed_before + self.car_speed) / 2 * self.dt
        arrival_num = len(np.where(self.car_pos > self.lane_length)[1])
        self.car_pos = self.car_pos[:, :-arrival_num]
        self.car_speed = self.car_speed[:, :-arrival_num]
        self.car_acc = self.car_acc[:, :-arrival_num]
