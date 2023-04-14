# -*- coding = uft-8 -*-
# @Time : 2023-03-24 16:21
# @Author : yzbyx
# @File : circle_frame.py
# @Software : PyCharm
from typing import Optional

import numpy as np

from trasim_simplified.core.constant import CFM
from trasim_simplified.core.frame.frame_abstract import FrameAbstract
from trasim_simplified.util.decorator.mydecorator import timer_no_log


class FrameCircle(FrameAbstract):
    def __init__(self, lane_length: int, car_num: int, car_length: int, car_initial_speed: int, speed_with_random: bool,
                 cf_mode: str, cf_param: dict[str, float]):
        super().__init__(lane_length, car_num, car_length, car_initial_speed, speed_with_random, cf_mode, cf_param)

    def car_init(self):
        dhw = self.lane_length / self.car_num
        assert dhw >= self.car_length, f"该密度下，车辆重叠！此车身长度下车辆数最多为{np.floor(self.lane_length / self.car_length)}"
        self.car_pos = np.arange(0, self.lane_length, dhw).reshape(1, -1)
        if self.car_num != self.car_pos.shape[1]:
            if (self.lane_length - self.car_pos[0][-1]) < 1e-6:
                self.car_pos = self.car_pos[:, :-1]
            Warning(f"车辆生成数量有误！目标：{self.car_num}，结果：{self.car_pos.shape}，头车位置：{self.car_pos[0][-1]}")
        if self.speed_with_random:
            self.car_speed = np.random.uniform(
                max(self.car_initial_speed - 0.5, 0),  self.car_initial_speed + 0.5, self.car_pos.shape
            ).reshape(1, -1)
        else:
            self.car_speed = (np.ones(self.car_pos.shape) * self.car_initial_speed).reshape(1, -1)
        self.car_acc = np.zeros(self.car_pos.shape).reshape(1, -1)

    def run(self, data_save=True, has_ui=True, **kwargs):
        return super().run(data_save, has_ui, **kwargs)

    def step(self):
        leader_x = np.roll(self.car_pos, -1)
        diff_x = leader_x - self.car_pos
        pos_ = np.where(diff_x < 0)
        leader_x[pos_] += self.lane_length
        self.car_acc = self.cf_model.step(
            self.car_speed,
            self.car_pos,
            np.roll(self.car_speed, -1),
            leader_x,
            self.car_length
        )


class FrameCircleW99(FrameCircle):
    """W99"""
    def __init__(self, lane_length: int, car_num: int, car_length: int, car_initial_speed: int, speed_with_random: bool,
                 cf_mode: str, cf_param: dict[str, float]):
        self.status: Optional[list] = None
        super().__init__(lane_length, car_num, car_length, car_initial_speed, speed_with_random, cf_mode, cf_param)

    def car_init(self):
        super().car_init()
        self.status = ["w",] * self.car_num

    def step(self):
        leader_x = np.roll(self.car_pos, -1)
        diff_x = leader_x - self.car_pos
        pos_ = np.where(diff_x < 0)
        leader_x[pos_] += self.lane_length
        leader_a = np.roll(self.car_acc, -1)
        leader_v = np.roll(self.car_speed, -1)
        for i in range(self.car_num):
            # interval, speed, acc, xOffset, length, leaderV, leaderA, leaderX, leaderL
            car_acc, status = self.cf_model.step(
                self.status[i],
                self.dt,
                self.car_speed[0][i],
                self.car_acc[0][i],
                self.car_pos[0][i],
                self.car_length,
                leader_v[0][i],
                leader_a[0][i],
                leader_x[0][i],
                self.car_length
            )
            self.status[i] = status
            self.car_acc[0][i] = car_acc
