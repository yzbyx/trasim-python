# -*- coding = uft-8 -*-
# @Time : 2023-03-24 16:21
# @Author : yzbyx
# @File : circle_frame.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.core.frame.frame_abstract import FrameAbstract
from trasim_simplified.msg.trasimWarning import TrasimWarning


class FrameCircle(FrameAbstract):
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

    def update_state(self):
        car_speed_before = self.car_speed.copy()
        self.car_speed += self.car_acc * self.dt

        speed_neg_pos = np.where(self.car_speed < 0)
        if len(speed_neg_pos[0]) != 0:
            TrasimWarning("存在速度为负的车辆！")
            self.car_speed[speed_neg_pos] = 0

        self.car_pos += (car_speed_before + self.car_speed) / 2 * self.dt
        self.car_pos[np.where(self.car_pos > self.lane_length)] -= self.lane_length


class FrameCircleCommon(FrameCircle):
    def _car_init(self):
        super()._car_init()
        self.status = [None,] * self.car_num

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
