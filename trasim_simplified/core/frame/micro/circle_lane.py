# -*- coding = uft-8 -*-
# @Time : 2023-03-24 16:21
# @Author : yzbyx
# @File : circle_lane.py
# @Software : PyCharm

from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class LaneCircle(LaneAbstract):
    def __init__(self, lane_length: float):
        super().__init__(lane_length)
        self.is_circle = True

    def step(self):
        for i, car in enumerate(self.car_list):
            car.step(i)

    def update_state(self):
        for car in self.car_list:
            self.car_state_update_common(car)

            car.x -= self.lane_length if car.x > self.lane_length else 0

        self.car_list = sorted(self.car_list, key=lambda c: c.x)
