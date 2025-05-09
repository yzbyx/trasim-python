# -*- coding: utf-8 -*-
# @time : 2023/5/12 10:32
# @Author : yzbyx
# @File : road.py
# Software: PyCharm
import random
import time
from typing import Optional

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

from trasim_simplified.core.constant import V_TYPE, MARKING_TYPE
from trasim_simplified.core.data.data_processor import DataProcessor
from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract
from trasim_simplified.core.frame.micro.open_lane import LaneOpen
from trasim_simplified.core.frame.micro.circle_lane import LaneCircle
from trasim_simplified.core.ui.pyqtgraph_ui import PyqtUI
from trasim_simplified.core.ui.sim_ui import UI2D
from trasim_simplified.core.data.data_container import Info as C_Info
from trasim_simplified.core.agent.vehicle import Vehicle
from trasim_simplified.core.ui.sim_ui_matplotlib import UI2DMatplotlib
from trasim_simplified.msg.trasimWarning import TrasimWarning


class Road:
    """在不影响Lane功能的基础上实现多车道道路"""
    def __init__(self, length: float, pyqtgraph=False):
        self.yield_ = True
        self.ID = 0
        self.lane_length = length
        self.lane_list: list[LaneAbstract] = []
        self.id_accumulate = 0

        if pyqtgraph:
            self.ui: PyqtUI | UI2DMatplotlib = PyqtUI(self)
        else:
            self.ui: PyqtUI | UI2DMatplotlib = UI2DMatplotlib(self)
        self.step_ = 0
        self.sim_step = None
        self.dt = None
        self.time_ = 0
        self.has_ui = False

        self.total_data = None

        self.data_processor: DataProcessor = DataProcessor()

        self._end_weaving_pos = None
        self._start_weaving_pos = None
        self.end_weaving_block_veh = Vehicle(None, V_TYPE.PASSENGER, -1, 0)

        self.mainline_end_indexes = []
        self.auxiliary_end_indexes = []

    @property
    def end_weaving_pos(self):
        return self._end_weaving_pos

    @property
    def start_weaving_pos(self):
        return self._start_weaving_pos

    @property
    def weaving_length(self):
        return self._end_weaving_pos - self._start_weaving_pos

    def set_end_weaving_pos(self, value):
        self.end_weaving_block_veh.x = value
        self._end_weaving_pos = value

    def set_start_weaving_pos(self, value):
        self._start_weaving_pos = value

    def add_lanes(self, lane_num: int, is_circle=False, lane_width_list: list[float] = None):
        """
        :param lane_width_list:
        :param lane_num:
        :param is_circle:
        :return:
        """
        real_index = list(range(lane_num))
        if lane_width_list is None:
            lane_width_list = [3.5] * lane_num
        for i in range(lane_num):
            if is_circle:
                lane = LaneCircle(self.lane_length, lane_width_list[i])
            else:
                lane = LaneOpen(self.lane_length, lane_width_list[i])
            lane.index = real_index[len(self.lane_list)]
            lane.add_num = len(self.lane_list)
            lane.ID = f"{lane.index}-{lane.add_num}"
            lane.road_control = True
            lane.road = self
            lane.left_neighbour_lane = None
            lane.right_neighbour_lane = None
            lane.update_y_info()
            self.lane_list.append(lane)
        for i, lane in enumerate(self.lane_list):
            for lane_ in self.lane_list:
                if lane.index - 1 == lane_.index:
                    lane.left_neighbour_lane = lane_
                elif lane.index + 1 == lane_.index:
                    lane.right_neighbour_lane = lane_
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
        data_head = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        time_stamp = "%s.%s" % (data_head, str(timeIn).split('.')[-1][:5])
        timeStart = time_stamp

        print("进入仿真主循环")
        for i, lane_iter in enumerate(lanes_iter):
            self.step_ = lane_iter.__next__()  # 车辆生成，数据记录

        for _ in tqdm.tqdm(range(self.sim_step)):
            # print("仿真步数：", self.step_)
            self.step_vehicle_info_update()  # 更新周围车辆信息
            self.step_vehicle_traj_pred()  # 轨迹预测
            if self.yield_: yield self.step_, 0  # 跟驰
            self.step_lc_intention_judge()
            if self.yield_: yield self.step_, 1  # 换道意图
            self.step_lc_decision_making()
            if self.yield_: yield self.step_, 2  # 换道决策
            self.step_vehicle_control()  # 控制量计算
            if self.yield_: yield self.step_, 3  # 控制量计算
            for i, lane_iter in enumerate(lanes_iter):
                self.step_ = lane_iter.__next__()  # 车辆控制量和坐标姿态更新
            if self.yield_: yield self.step_, 4
            self.step_vehicle_lane_update()  # 车辆车道更新和前后车辆更新
            # print("车辆车道更新完成")
            for i, lane_iter in enumerate(lanes_iter):
                self.step_ = lane_iter.__next__()  # 车辆输入与数据记录
            self.step_ += 1
            self.time_ += self.dt
            if self.has_ui: self.ui.ui_update()

        timeOut = time.time()
        log_string = '[' + self.run.__name__ + '] ' + 'time usage: ' + timeStart + ' + ' + \
                     str((timeOut - timeIn) * 1000 // 1 / 1000) + ' s'
        print(log_string)

    def step_lc_intention_judge(self):
        for i, lane in enumerate(self.lane_list):
            for j, car in enumerate(lane.car_list):
                if car.type != V_TYPE.OBSTACLE and not car.skip:
                    car.lc_intention_judge()
                    assert car.target_lane is not None

    def step_lc_decision_making(self):
        for i, lane in enumerate(self.lane_list):
            for j, car in enumerate(lane.car_list):
                if car.type != V_TYPE.OBSTACLE and not car.skip:
                    car.lc_decision_making()
                    assert car.target_lane is not None

    def step_vehicle_control(self):
        for i, lane in enumerate(self.lane_list):
            for j, car in enumerate(lane.car_list):
                if car.type != V_TYPE.OBSTACLE and not car.skip:
                    car.cal_vehicle_control()
                    assert car.target_lane is not None

    def step_vehicle_lane_update(self):
        for i, lane in enumerate(self.lane_list):
            for j, car in enumerate(lane.car_list):
                if car.type != V_TYPE.OBSTACLE:
                    if car.y > lane.y_left:
                        target_lane = lane.left_neighbour_lane
                        car.lane.car_remove(car)
                        target_lane.car_insert_by_instance(car)
                        car.lane = target_lane
                    elif car.y < lane.y_right:
                        target_lane = lane.right_neighbour_lane
                        car.lane.car_remove(car)
                        target_lane.car_insert_by_instance(car)
                        car.lane = target_lane
                    assert car.target_lane is not None

    def step_vehicle_info_update(self):
        for i, lane in enumerate(self.lane_list):
            for j, car in enumerate(lane.car_list):
                if car.type != V_TYPE.OBSTACLE:
                    car.update_necessary_info()

    def step_vehicle_traj_pred(self):
        for i, lane in enumerate(self.lane_list):
            for j, car in enumerate(lane.car_list):
                if car.type != V_TYPE.OBSTACLE:
                    car.cal_traj_pred()

    def get_lane_index(self, y):
        """
        获取车道编号
        :param y:
        :return:
        """
        for i, lane in enumerate(self.lane_list):
            if lane.y_right <= y < lane.y_left:
                return i
        return None

    def choose_vehicle(self, lane_add_num: int = 0):
        """
        随机选取一辆车辆
        :param lane_add_num:
        :return:
        """
        all_vehicles = []
        for lane in self.lane_list:
            if lane.add_num == lane_add_num:
                all_vehicles.extend(lane.car_list)
        if len(all_vehicles) == 0:
            TrasimWarning("没有车辆！")
            return None
        return random.choice(all_vehicles)

    @staticmethod
    def get_neighbour_vehicles(car: Vehicle, type_="all"):
        """
        获取指定车道的相邻车辆
        :param car:
        :param type_: 车辆类型, all: 所有车辆, left: 左侧车辆, right: 右侧车辆
        :return:
        """
        left_front = left_rear = None
        right_front = right_rear = None
        left_lane, right_lane = Road.get_available_adjacent_lane(car.lane, car.x)
        if type_ == "left" or type_ == "all":
            lane = left_lane
            if lane is not None:
                left_rear, left_front = lane.get_relative_car(car)
            if type_ == "left":
                return left_front, left_rear
        if type_ == "right" or type_ == "all":
            lane = right_lane
            if lane is not None:
                right_rear, right_front = lane.get_relative_car(car)
            if type_ == "right":
                return right_front, right_rear
        return left_rear, left_front, right_rear, right_front

    @staticmethod
    def _check_and_correct_lc_pos(target_lane, car_lc_last, car):
        """
        对可能的多车换到同一车道的检查
        :param target_lane:
        :param car_lc_last:
        :param car:
        :return:
        """
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

    def get_appropriate_car(self, lane_add_num=0):
        return self.lane_list[lane_add_num].get_appropriate_car()

    def get_car_info(self, car_id: int, info: str, lane_add_num=None):
        if lane_add_num is None:
            for lane in self.lane_list:
                result = lane.get_car_info(car_id, info)
                if result is not None:
                    return result
        else:
            result = self.lane_list[lane_add_num].get_car_info(car_id, info)
            if result is not None:
                return result

        TrasimWarning("未找到车辆！")

    def car_insert_middle(self, lane_add_num: int = 0, *args, **kwargs):
        return self.lane_list[lane_add_num].car_insert_middle(*args, **kwargs)

    @staticmethod
    def get_available_adjacent_lane(lane: LaneAbstract, x_lon) -> \
            tuple[Optional[LaneAbstract], Optional[LaneAbstract]]:
        left, right = lane.left_neighbour_lane, lane.right_neighbour_lane
        left_type, right_type = lane.get_marking_type(x_lon)

        if MARKING_TYPE.SOLID == left_type:
            left = None

        if MARKING_TYPE.SOLID == right_type:
            right = None

        return left, right

    def data_to_df(self):
        if self.total_data is None:
            self.total_data = pd.concat([lane.data_container.data_to_df() for lane in self.lane_list], axis=0,
                                        ignore_index=True)
            self.total_data = self.total_data.sort_values(by=[C_Info.lane_add_num, C_Info.id, C_Info.step])\
                .reset_index(drop=True)
        assert len(self.total_data) != 0, "数据为空！"
        return self.total_data

    def find_on_lanes(self, car_id: int):
        df = self.data_to_df()
        return df[df[C_Info.id] == car_id][C_Info.lane_add_num].unique().tolist()

    def take_over(self, car_id: int, acc_values: float, lc_result: dict):
        """控制指定车辆运动"""
        for lane in self.lane_list:
            for car in lane.car_list:
                if car.ID == car_id:
                    car.cf_acc = acc_values
                    car.lc_result = lc_result

    def draw(self, ax: plt.Axes, fill=True):
        """绘制车道线"""
        static_lines = []
        for i, lane in enumerate(self.lane_list):
            x = np.linspace(0, self.lane_length, 2)
            # y = np.ones_like(x) * lane.y_center
            # ax.plot(x, y, color="white", linewidth=0.5)
            if fill:
                ax.fill_between(x, lane.y_left, lane.y_right, color="gray", alpha=0.5)

            if lane.marking_type is not None:
                for j in range(len(lane.marking_type[0]) - 1):
                    x = np.linspace(lane.marking_type[0][j], lane.marking_type[0][j + 1], 2)
                    for k, y_marking in enumerate(lane.marking_type[1][j]):
                        if k == 0:
                            if lane.index != 0:
                                continue
                            else:
                                y = np.ones_like(x) * lane.y_left
                        else:
                            y = np.ones_like(x) * lane.y_right

                        if y_marking == MARKING_TYPE.SOLID:
                            if fill:
                                color = "yellow"
                            else:
                                color = "orange"
                            line = ax.plot(x, y, color=color, linewidth=1)[-1]
                        elif y_marking == MARKING_TYPE.DASHED:
                            if fill:
                                color = "white"
                            else:
                                color = "black"
                            line = ax.plot(x, y, color=color, linestyle="--", linewidth=1)[-1]
                        else:
                            raise ValueError(f"Unknown marking type: {y_marking}")

                        static_lines.append(line)
        return static_lines

    def reset(self):
        """清理数据"""
        self.total_data = None
        self.id_accumulate = 0
        for lane in self.lane_list:
            lane.reset()
        self.sim_step = 0
        self.step_ = 0
        self.time_ = 0
        self.dt = None
        self.has_ui = False
        self.yield_ = True
