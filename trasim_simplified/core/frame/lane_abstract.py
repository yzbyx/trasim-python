# -*- coding = uft-8 -*-
# @Time : 2023-03-25 22:37
# @Author : yzbyx
# @File : frame.py
# @Software : PyCharm
import abc
from abc import ABC

import numpy as np

from trasim_simplified.core.data.data_container import DataContainer
from trasim_simplified.core.data.data_processor import DataProcessor
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.ui.sim_ui import UI
from trasim_simplified.core.vehicle import Vehicle


class LaneAbstract(ABC):
    def __init__(self, lane_length: int):
        self.car_num_total = 0
        self.is_circle = None
        self.lane_length = float(lane_length)

        self.id_accumulate = 0
        self.car_num_list: list[int] = []
        self.car_type_list: list[str] = []
        self.car_length_list: list[float] = []
        self.car_initial_speed_list: list[float] = []
        self.speed_with_random_list: list[bool] = []
        self.cf_name_list: list[str] = []
        self.cf_param_list: list[dict] = []

        self.car_list: list[Vehicle] = []
        self.out_car_has_data: list[Vehicle] = []

        self.step_ = 0
        """当前仿真步次"""
        self.time_ = 0
        """当前仿真时长 [s]"""
        self.yield_ = True
        """run()是否为迭代器"""

        self.dt = 0.1
        """仿真步长 [s]"""
        self.warm_up_step = int(5 * 60 / self.dt)
        """预热时间 [s]"""
        self.sim_step = int(10 * 60 / self.dt)
        """总仿真步 [次]"""

        self.data_save = False
        self.data_container: DataContainer = DataContainer(self)
        self.data_processor: DataProcessor = DataProcessor(self)

        self.plot: Plot = Plot(self)

        self.has_ui = False
        self.ui: UI = UI(self)

    def _get_new_car_id(self):
        self.id_accumulate += 1
        return self.id_accumulate

    @property
    def car_num(self):
        return len(self.car_list)

    def car_config(self, car_num: int, car_length: float, car_type: str, car_initial_speed: int, speed_with_random: bool,
                   cf_name: str, cf_param: dict[str, float]):
        """如果是开边界，则car_num与car_loader配合可以代表车型比例，如果car_loader中的flow为复数，则car_num为真实生成车辆数"""
        self.car_num_list.append(car_num)
        self.car_length_list.append(car_length)
        self.car_type_list.append(car_type)
        self.car_initial_speed_list.append(car_initial_speed)
        self.speed_with_random_list.append(speed_with_random)
        self.cf_name_list.append(cf_name)
        self.cf_param_list.append(cf_param)

    def car_load(self, car_gap=-1):
        car_num_total = sum(self.car_num_list)
        car_length_total = np.sum(np.array(self.car_num_list) * np.array(self.car_length_list))
        gap = (self.lane_length - car_length_total) / car_num_total
        assert gap >= 0, f"该密度下，车辆重叠！"

        index_list = np.arange(car_num_total)
        np.random.shuffle(index_list)

        x = 0
        car_count = 0
        for i, car_num in enumerate(self.car_num_list):
            for j in range(car_num):
                vehicle = Vehicle(self, self.car_type_list[i], self._get_new_car_id(), self.car_length_list[i])
                vehicle.x = x
                vehicle.v = np.random.uniform(
                    max(self.car_initial_speed_list[i] - 0.5, 0), self.car_initial_speed_list[i] + 0.5
                ) if self.speed_with_random_list[i] else self.car_initial_speed_list[i]
                vehicle.a = 0
                vehicle.set_cf_model(self.cf_name_list[i], self.cf_param_list[i])

                self.car_list.append(vehicle)
                if car_count != car_num_total - 1:
                    if car_gap < 0:
                        if j < car_num - 1:
                            x = x + gap + self.car_length_list[i]
                        else:
                            x = x + gap + self.car_length_list[i + 1]
                    else:
                        x = x + car_gap + self.car_length_list[car_count + 1]
                car_count += 1

        for i, car in enumerate(self.car_list[1: -1]):
            car.leader = self.car_list[i + 2]
            car.follower = self.car_list[i]
        if self.is_circle is True:
            self.car_list[0].leader = self.car_list[1]
            self.car_list[0].follower = self.car_list[-1]
            self.car_list[-1].follower = self.car_list[-2]
            self.car_list[-1].leader = self.car_list[0]

    def run(self, data_save=True, has_ui=True, **kwargs):
        self.data_save = data_save
        self.has_ui = has_ui

        if kwargs is None:
            kwargs = {}
        self.dt = kwargs.get("dt", 0.1)
        """仿真步长 [s]"""
        self.warm_up_step = kwargs.get("warm_up_step", int(5 * 60 / self.dt))
        """预热步数 [s]"""
        self.sim_step = kwargs.get("sim_step", int(10 * 60 / self.dt))
        """总仿真步 [次]"""
        frame_rate = kwargs.get("frame_rate", -1)
        """pygame刷新率 [fps]"""
        caption = kwargs.get("ui_caption", "微观交通流仿真")
        self.yield_ = kwargs.get("if_yield", True)
        """run()是否为迭代器"""

        if has_ui:
            self.ui.ui_init(caption=caption, frame_rate=frame_rate)

        # 整个仿真能够运行sim_step的仿真步
        while self.sim_step != self.step_:
            if not self.is_circle:
                self.car_summon()
            # 能够记录warm_up_step仿真步时的车辆数据
            if self.data_save and self.step_ >= self.warm_up_step:
                self.record()
            self.step()  # 未更新状态，但已经计算
            # 控制车辆对应的step需要在下一个仿真步才能显现到数据记录中
            if self.yield_: yield self.step_
            self.update_state()  # 更新车辆状态
            self.step_ += 1
            self.time_ += self.dt
            if self.has_ui: self.ui.ui_update()

    @abc.abstractmethod
    def update_state(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    def car_summon(self):
        """用于开边界车辆生成"""
        pass

    def record(self):
        for car in self.car_list:
            car.record()

    def get_appropriate_car(self) -> int:
        """获取合适进行控制扰动的单个车辆，（未到车道一半线的最近车辆）"""
        pos = self.lane_length / 2
        car_pos = np.array([car.x for car in self.car_list])
        pos_ = np.where(car_pos < pos)[0]
        max_pos = np.argmax(car_pos[pos_])
        return self.car_list[pos_[max_pos]].ID

    def take_over(self, car_id: int, acc_values: float):
        """控制指定车辆运动"""
        for car in self.car_list:
            if car.ID == car_id:
                car.step_acc = acc_values

    def get_relative_id(self, id_, offset: int):
        """
        :param id_: 车辆ID
        :param offset: 正整数代表向下游检索，负数代表上游
        """
        assert offset - int(offset) == 0, "offset必须是整数"
        for car in self.car_list:
            if car.ID == id_:
                while offset != 0:
                    if offset > 0:
                        if car.leader is not None:
                            car = car.leader
                        offset -= 1
                    else:
                        if car.follower is not None:
                            car = car.follower
                        offset += 1
                return car.ID

    def __str__(self):
        return "lane_length: " + str(self.lane_length) + \
            "\tcar_num: " + str(self.car_num_list) + \
            "\tcar_length: " + str(self.car_length_list) + \
            "\tcar_initial_speed: " + str(self.car_initial_speed_list) + \
            "\tbasic_record: " + str(self.data_save) + \
            "\thas_ui: " + str(self.has_ui) + \
            "\tframe_rate" + str(self.ui.frame_rate) + \
            "\tdt: " + str(self.dt) + \
            "\twarm_up_step: " + str(self.warm_up_step) + \
            "\tsim_step: " + str(self.sim_step)
