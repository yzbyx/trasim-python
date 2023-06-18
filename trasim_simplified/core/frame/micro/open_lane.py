# -*- coding = uft-8 -*-
# @Time : 2023-04-14 12:07
# @Author : yzbyx
# @File : open_lane.py
# @Software : PyCharm
from typing import Optional

import numpy as np

from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract
from trasim_simplified.core.vehicle import Vehicle


class THW_DISTRI:
    Uniform = "uniform"
    Exponential = "exponential"


class LaneOpen(LaneAbstract):
    def __init__(self, lane_length: float):
        super().__init__(lane_length)
        self.is_circle = False
        self.outflow_point = True
        """此车道是否为流出车道 (影响车辆跟驰行为)"""
        self.flow_rate = -1
        """流入流率(veh/s) (负数则看car_config配置)"""
        self.thw_distri = THW_DISTRI.Uniform
        """车辆到达时间表，默认均匀分布"""
        np.random.seed(0)
        self.car_num_percent: Optional[np.ndarray] = None
        self.next_car_time = 0
        self.fail_summon_num = 0
        self.offset_pos = 0

    def car_loader(self, flow_rate: int | float, thw_distribution: str = THW_DISTRI.Uniform,
                   offset_time: float = 0, offset_pos: float = 0):
        """
        车辆生成器配置

        :param offset_pos: 车辆生成点距离车道起点位置的偏移 [m]
        :param offset_time: 流量生成延迟 [s]
        :param thw_distribution: 车辆到达分布，None则默认均匀分布
        :param flow_rate: 总流量 (veh/h)
        """
        self.flow_rate = flow_rate / 3600
        self.thw_distri = thw_distribution
        self.next_car_time += offset_time
        self.offset_pos = offset_pos
        self.car_num_percent = np.array(self.car_num_list) / sum(self.car_num_list)

    def step(self):
        for i, car in enumerate(self.car_list):
            car.step(i)

    def car_summon(self):
        if 0 < self.time_ < self.next_car_time:
            return
        if self.next_car_time <= self.time_:
            # 车辆类别随机
            assert self.car_num_percent is not None
            if len(self.car_list) != 0:
                first = self.car_list[0]
                # if first.x - first.length - first.v * self.dt < 0:
                if first.x - first.length < 0:
                    self.fail_summon_num += 1
                    print(f"车道{self.ID}生成车辆失败！共延迟{self.fail_summon_num}个仿真步")
                    return

            i = np.random.choice(self.car_num_percent, p=self.car_num_percent.ravel())
            pos = np.where(self.car_num_percent == i)[0]
            if len(pos) > 1:
                i = np.random.choice(pos)
            else:
                i = pos[0]
            vehicle = Vehicle(self, self.car_type_list[i], self._get_new_car_id(), self.car_length_list[i])
            vehicle.x = self.offset_pos
            vehicle.set_cf_model(self.cf_name_list[i], self.cf_param_list[i])
            vehicle.set_lc_model(self.lc_name_list[i], self.lc_param_list[i])
            if self.car_initial_speed_list[i] >= 0:
                vehicle.v = np.random.uniform(
                    max(self.car_initial_speed_list[i] - 0.5, 0), self.car_initial_speed_list[i] + 0.5
                ) if self.speed_with_random_list[i] else self.car_initial_speed_list[i]
            else:
                if len(self.car_list) == 0:
                    vehicle.v = vehicle.cf_model.get_expect_speed()
                else:
                    vehicle.v = self.car_list[0].v
            vehicle.a = 0
            vehicle.set_car_param(self.car_param_list[i])

            if len(self.car_list) != 0:
                vehicle.leader = self.car_list[0]
                self.car_list[0].follower = vehicle

            self.car_list.insert(0, vehicle)

            self._set_next_summon_time()

    def _set_next_summon_time(self):
        #  车头时距随机
        if self.thw_distri == THW_DISTRI.Uniform:
            if self.flow_rate > 0:
                thw = 1 / self.flow_rate
            else:
                thw = np.inf
        elif self.thw_distri == THW_DISTRI.Exponential:
            a = np.random.random()
            if self.flow_rate > 0:
                thw = - np.log(a) / self.flow_rate
            else:
                thw = np.inf
        else:
            if self.flow_rate > 0:
                thw = 1 / self.flow_rate
            else:
                thw = np.inf
        self.next_car_time += thw

    def update_state(self):
        for car in self.car_list:
            self.car_state_update_common(car)

            if car.x > self.lane_length:
                self.car_remove(car, car.has_data())


class CarLoader:
    pass
