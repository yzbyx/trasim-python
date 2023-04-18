# -*- coding = uft-8 -*-
# @Time : 2023-03-25 22:37
# @Author : yzbyx
# @File : frame.py
# @Software : PyCharm
import abc
from abc import ABC
from typing import Optional, Union

import numpy as np

from trasim_simplified.core.data.data_container import DataContainer
from trasim_simplified.core.data.data_processor import DataProcessor
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.ui.sim_ui import UI
from trasim_simplified.core.kinematics.cfm import get_cf_model
from trasim_simplified.msg.trasimWarning import TrasimWarning


class FrameAbstract(ABC):
    def __init__(self, lane_length: int, car_num: int, car_length: int, car_initial_speed:int, speed_with_random: bool,
                 cf_mode: str, cf_param: dict[str, float]):
        self.car_num = int(car_num)
        self.car_length = float(car_length)
        self.lane_length = float(lane_length)
        self.car_initial_speed = float(car_initial_speed)
        self.speed_with_random = speed_with_random
        self.cf_mode = cf_mode
        self.cf_param = cf_param

        self.cf_model = get_cf_model(None, cf_mode, cf_param)
        self.cf_param = self.cf_model.get_param_map()

        self.car_pos: Optional[np.ndarray] = None
        self.car_speed: Optional[np.ndarray] = None
        self.car_acc: Optional[np.ndarray] = None
        self._car_init()
        assert self.car_pos is not None, "car_init()函数未初始化car_pos属性!"
        assert self.car_speed is not None, "car_init()函数未初始化car_speed属性!"
        assert self.car_acc is not None, "car_init()函数未初始化car_acc属性!"

        self.step_ = 0
        """当前仿真步次"""
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

    def _car_init(self):
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
        self.car_id = np.arange(self.car_num, 0, -1).reshape(1, -1)

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

        if has_ui: self.ui.ui_init(caption=caption, frame_rate=frame_rate)

        # 整个仿真能够运行sim_step的仿真步
        while self.sim_step != self.step_:
            # 能够记录warm_up_step仿真步时的车辆数据
            if self.data_save and self.step_ >= self.warm_up_step:
                self.data_container.record()
            self.step()
            # 控制车辆对应的step需要在下一个仿真步才能显现到数据记录中
            if self.yield_: yield self.step_
            self.update_state()
            self.step_ += 1
            if self.has_ui: self.ui.ui_update()

    @abc.abstractmethod
    def update_state(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    def get_appropriate_car(self) -> int:
        """获取合适进行控制扰动的单个车辆，（未到车道一半线的最近车辆）"""
        pos = self.lane_length / 2
        pos_ = np.where(self.car_pos < pos)
        max_pos = np.argmax(self.car_pos[pos_])
        return pos_[1][max_pos]

    def take_over(self, car_indexes: Union[int, list, tuple, np.ndarray],
                  acc_values: Union[int, float, list, tuple, np.ndarray]):
        """控制指定车辆运动"""
        car_indexes = np.array(car_indexes)
        acc_values = np.array(acc_values)
        assert car_indexes.shape == acc_values.shape, "传入参数形状不同！"

        pos = (0, car_indexes)
        self.car_acc[pos] = acc_values

    def __str__(self):
        return "lane_length: " + str(self.lane_length) + \
            "\tcar_num: " + str(self.car_num) + \
            "\tcar_length: " + str(self.car_length) + \
            "\tcar_initial_speed: " + str(self.car_initial_speed) + \
            "\tcf_mode: " + self.cf_mode + \
            "\tcf_param: " + self.cf_model.get_param_map().__str__() + \
            "\tbasic_record: " + str(self.data_save) + \
            "\thas_ui: " + str(self.has_ui) + \
            "\tframe_rate" + str(self.ui.frame_rate) + \
            "\tdt: " + str(self.dt) + \
            "\twarm_up_step: " + str(self.warm_up_step) + \
            "\tsim_step: " + str(self.sim_step)
