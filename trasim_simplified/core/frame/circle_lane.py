# -*- coding = uft-8 -*-
# @Time : 2023-03-24 16:21
# @Author : yzbyx
# @File : circle_lane.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.core.frame.lane_abstract import LaneAbstract
from trasim_simplified.msg.trasimWarning import TrasimWarning


class LaneCircle(LaneAbstract):
    def __init__(self, lane_length: int):
        super().__init__(lane_length)
        self.is_circle = True

    def step(self):
        for i, car in enumerate(self.car_list):
            car.step(i)

    def update_state(self):
        for car in self.car_list:
            car_speed_before = car.v
            car.v += car.step_acc * self.dt
            car.a = car.step_acc

            if car.v < 0:
                TrasimWarning("存在速度为负的车辆！")
                car.v = 0

            car.x += (car_speed_before + car.v) / 2 * self.dt
            car.x -= self.lane_length if car.x > self.lane_length else 0

        self.car_list = sorted(self.car_list, key=lambda c: c.x)
