# -*- coding: utf-8 -*-
# @Time : 2023/5/12 10:32
# @Author : yzbyx
# @File : road.py
# Software: PyCharm
import time
from typing import Optional

import pandas as pd

from trasim_simplified.core.constant import SECTION_TYPE, V_TYPE
from trasim_simplified.core.data.data_processor import DataProcessor
from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract
from trasim_simplified.core.frame.micro.open_lane import LaneOpen
from trasim_simplified.core.frame.micro.circle_lane import LaneCircle
from trasim_simplified.core.ui.sim_ui import UI
from trasim_simplified.core.data.data_container import Info as C_Info
from trasim_simplified.util.decorator.timer import _get_current_time


class Road:
    """在不影响Lane功能的基础上实现多车道道路"""
    def __init__(self, length: float):
        self.yield_ = True
        self.ID = 0
        self.lane_length = length
        self.lane_list: list[LaneAbstract] = []
        self.id_accumulate = 0

        self.ui: UI = UI(self)
        self.step_ = 0
        self.sim_step = None
        self.dt = None
        self.time_ = 0
        self.has_ui = False

        self.total_data = None

        self.data_processor: DataProcessor = DataProcessor()

    def add_lanes(self, lane_num: int, is_circle=True):
        for i in range(lane_num):
            if is_circle:
                lane = LaneCircle(self.lane_length)
            else:
                lane = LaneOpen(self.lane_length)
            lane.ID = len(self.lane_list)
            lane.index = len(self.lane_list)
            lane.road_control = True
            lane.road = self
            self.lane_list.append(lane)
        return self.lane_list

    def run(self, data_save=True, has_ui=True, **kwargs):
        kwargs.update({"yield": True})
        self.dt = kwargs.get("dt", 0.1)
        """仿真步长 [s]"""
        self.sim_step = kwargs.get("sim_step", int(10 * 60 / self.dt))
        """总仿真步 [次]"""
        self.has_ui = has_ui

        if self.has_ui:
            self.ui.ui_init(frame_rate=kwargs.get("frame_rate", -1))

        lanes_iter = [lane.run(data_save=data_save, has_ui=False, **kwargs) for lane in self.lane_list]

        timeIn = time.time()
        timeStart = _get_current_time()

        while self.sim_step != self.step_:
            for i, lane_iter in enumerate(lanes_iter):
                self.step_ = lane_iter.__next__()
            if self.yield_: yield self.step_, 0  # 跟驰
            for i, lane_iter in enumerate(lanes_iter):
                self.step_ = lane_iter.__next__()  # 跟驰状态更新
            self.step_lane_change()
            if self.yield_: yield self.step_, 1  # 换道
            self.update_lc_state()  # 换道状态更新
            self.step_ += 1
            self.time_ += self.dt
            if self.has_ui: self.ui.ui_update()

        timeOut = time.time()
        log_string = '[' + self.run.__name__ + '] ' + 'time usage: ' + timeStart + ' + ' + \
                     str((timeOut - timeIn) * 1000 // 1 / 1000) + ' s'
        print(log_string)

    def step_lane_change(self):
        for i, lane in enumerate(self.lane_list):
            for j, car in enumerate(lane.car_list):
                if car.type != V_TYPE.OBSTACLE:
                    left, right = self.get_available_adjacent_lane(i, car.x)
                    car.step_lane_change(j, left, right)

    def update_lc_state(self):
        for i, lane in enumerate(self.lane_list):
            car_lc_last = None
            car_list = lane.car_list.copy()
            car_list.reverse()
            for j, car in enumerate(car_list):
                if car.type != V_TYPE.OBSTACLE:
                    lc = car.lc_result.get("lc", 0)
                    if lc != 0:
                        target_lane = self.lane_list[car.lane.index + lc]
                        if self._check_and_correct_lc_pos(target_lane, car_lc_last, car):
                            lane.car_remove(car)
                            car.x = car.lc_result.get("x", car.x)
                            car.v = car.lc_result.get("v", car.v)
                            car.a = car.lc_result.get("a", car.a)
                            target_lane.car_insert_by_instance(car)
                            car_lc_last = car
                        car.lc_result = {"lc": 0}

    @staticmethod
    def _check_and_correct_lc_pos(target_lane, car_lc_last, car):
        target_pos = car.lc_result.get("x", car.x)
        if target_lane.is_circle and target_pos > target_lane.lane_length:
            car.lc_result["x"] -= target_lane.lane_length
        if car_lc_last is None or car_lc_last.lane != target_lane:
            return True
        dist = car_lc_last.get_dist(target_pos)
        if dist > car.length or dist < - car_lc_last.length:
            return True
        return False

    def get_new_car_id(self):
        self.id_accumulate += 1
        return self.id_accumulate

    def get_appropriate_car(self, lane_index=0):
        return self.lane_list[lane_index].get_appropriate_car()

    def get_available_adjacent_lane(self, lane_index, pos) -> tuple[Optional[LaneAbstract], Optional[LaneAbstract]]:
        lane = self.lane_list[lane_index]
        left, right = self._get_adjacent_lane(lane_index)
        section_type = lane.get_section_type(pos)
        if SECTION_TYPE.NO_LEFT in section_type:
            left = None
        if SECTION_TYPE.NO_RIGHT in section_type:
            right = None
        return left, right

    def _get_adjacent_lane(self, index: int) -> tuple[Optional[LaneAbstract], Optional[LaneAbstract]]:
        assert len(self.lane_list) > 0
        if len(self.lane_list) == 1:
            return None, None
        if index == 0:
            return None, self.lane_list[1]
        elif index == len(self.lane_list) - 1:
            return self.lane_list[-2], None
        else:
            return self.lane_list[index - 1], self.lane_list[index + 1]

    def data_to_df(self):
        if self.total_data is None:
            self.total_data = pd.concat([lane.data_container.data_to_df() for lane in self.lane_list], axis=0,
                                        ignore_index=True)
            self.total_data = self.total_data.sort_values(by=[C_Info.lane_id, C_Info.id, C_Info.step])\
                .reset_index(drop=True)
        return self.total_data

    def find_on_lanes(self, car_id: int):
        df = self.data_to_df()
        return df[df[C_Info.id] == car_id][C_Info.lane_id].unique().tolist()

    def take_over(self, car_id: int, acc_values: float, lc_result: dict):
        """控制指定车辆运动"""
        for lane in self.lane_list:
            for car in lane.car_list:
                if car.ID == car_id:
                    car.cf_acc = acc_values
                    car.lc_result = lc_result
