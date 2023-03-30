# -*- coding = uft-8 -*-
# @Time : 2023-03-25 22:37
# @Author : yzbyx
# @File : frame_abstract.py
# @Software : PyCharm
import abc
from abc import ABC
from typing import Optional, Iterable, Union

import numpy as np
import pandas as pd
import pygame as pg
from matplotlib import pyplot as plt
import matplotlib.colors as mc
import matplotlib.collections as mcoll
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from pygame.time import Clock

from trasim_simplified.kinematics.cfm import get_cf_model
from trasim_simplified.msg.trasimWarning import TrasimWarning


class FrameAbstract(ABC):
    def __init__(self, lane_length: int, car_num: int, car_length: int, car_initial_speed:int,
                 cf_mode: str, cf_param: dict[str, float]):
        self.car_num = car_num
        self.car_length = car_length
        self.lane_length = lane_length
        self.car_initial_speed = car_initial_speed
        self.cf_mode = cf_mode
        self.cf_param = cf_param

        self.cf_model = get_cf_model(None, cf_mode, cf_param)
        self.cf_param = self.cf_model.get_param_map()

        self.car_pos: Optional[np.ndarray] = None
        self.car_speed: Optional[np.ndarray] = None
        self.car_acc: Optional[np.ndarray] = None
        self.car_init()
        assert self.car_pos is not None, "car_init()函数未初始化car_pos属性!"
        assert self.car_speed is not None, "car_init()函数未初始化car_speed属性!"
        assert self.car_acc is not None, "car_init()函数未初始化car_acc属性!"

        self.step_ = 0
        """当前仿真时间 [s]"""

        self.dt = 0.1
        """仿真步长 [s]"""
        self.warm_up_step = int(5 * 60 / self.dt)
        """预热时间 [s]"""
        self.sim_step = int(10 * 60 / self.dt)
        """总仿真步 [次]"""

        self.acc_data: Optional[np.ndarray] = None
        self.speed_data: Optional[np.ndarray] = None
        self.pos_data: Optional[np.ndarray] = None
        self.gap_data: Optional[np.ndarray] = None
        self.dhw_data: Optional[np.ndarray] = None
        self.thw_data: Optional[np.ndarray] = None
        self.dv_data: Optional[np.ndarray] = None

        self.basic_save = False
        self.aggregate_cal = False
        self.plot_data = False
        self.df_save = False

        self.ui = False
        self.frame_rate = -1
        self.screen_width = 1980 / 1.5
        self.screen_height = 100 / 1.5
        self.screen: Optional[pg.Surface] = None
        self.clock: Optional[Clock] = None

    @abc.abstractmethod
    def car_init(self):
        pass

    def run(self, basic_save=True, aggregate_cal=True, plot_data=True, df_save=True, ui=True, **kwargs):
        self.basic_save = basic_save
        self.aggregate_cal = aggregate_cal
        self.plot_data = plot_data
        self.ui = ui
        self.df_save = df_save

        if kwargs is None:
            kwargs = {}
        self.dt = kwargs.get("dt", 0.1)
        """仿真步长 [s]"""
        self.warm_up_step = kwargs.get("warm_up_step", int(5 * 60 / self.dt))
        """预热步数 [s]"""
        self.sim_step = kwargs.get("sim_step", int(10 * 60 / self.dt))
        """总仿真步 [次]"""
        self.frame_rate = kwargs.get("frame_rate", -1)
        """pygame刷新率 [fps]"""

        if ui: self.ui_init()

        while self.sim_step != self.step_:
            if self.basic_save and self.step_ > self.warm_up_step:
                self.record()
            self.step()
            yield self.step_
            self.update_state()
            self.step_ += 1
            if self.ui: self.ui_update()

        if self.aggregate_cal and self.basic_save:
            print(self.aggregate())
            if self.plot_data: self.plot(kwargs.get("plot_index", 0) )
            if self.df_save: self.data_to_df()

    def update_state(self):
        car_speed_before = self.car_speed.copy()
        self.car_speed += self.car_acc * self.dt

        speed_neg_pos = np.where(self.car_speed < 0)
        if len(speed_neg_pos[0]) != 0:
            TrasimWarning("存在速度为负的车辆！")
            self.car_speed[speed_neg_pos] = 0

        self.car_pos += (car_speed_before + self.car_speed) / 2 * self.dt
        self.car_pos[np.where(self.car_pos > self.lane_length)] -= self.lane_length

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

    def _get_dhw(self, car_pos):
        car_pos = np.concatenate([car_pos, [[car_pos[0, 0]]]], axis=1)
        dhw = np.diff(car_pos)
        dhw[np.where(dhw < 0)] += self.lane_length
        return dhw

    @staticmethod
    def _get_dv(car_speed):
        car_speed = np.concatenate([car_speed, [[car_speed[0, 0]]]], axis=1)
        return - np.diff(car_speed)

    def record(self):
        self.acc_data = np.concatenate([self.acc_data, self.car_acc], axis=0) \
            if self.acc_data is not None else self.car_acc.copy()
        self.speed_data = np.concatenate([self.speed_data, self.car_speed], axis=0) \
            if self.speed_data is not None else self.car_speed.copy()
        self.pos_data = np.concatenate([self.pos_data, self.car_pos], axis=0) \
            if self.pos_data is not None else self.car_pos.copy()

        self.extend_record()

    def extend_record(self):
        current_dhw = self._get_dhw(self.car_pos)
        self.dhw_data = np.concatenate([self.dhw_data, current_dhw], axis=0) \
            if self.dhw_data is not None else current_dhw
        self.gap_data = np.concatenate([self.gap_data, current_dhw - self.car_length], axis=0) \
            if self.gap_data is not None else current_dhw - self.car_length
        self.thw_data = np.concatenate([self.thw_data, current_dhw / (self.car_speed + np.finfo(np.float32).eps)], axis=0) \
            if self.thw_data is not None else current_dhw / self.car_speed
        self.dv_data = np.concatenate([self.dv_data, self._get_dv(self.car_speed)], axis=0) \
            if self.dv_data is not None else self._get_dv(self.car_speed)

    def aggregate(self):
        assert self.speed_data.size != 0 and self.gap_data.size != 0 and self.dhw_data.size != 0 and \
               self.thw_data.size != 0 and self.dv_data.size != 0 and self.pos_data.size != 0, \
            "调用本函数须使用record函数记录数据"
        avg_speed, avg_gap, avg_dv, avg_dhw, avg_thw = \
            (np.mean(data) for data in [self.speed_data, self.gap_data, self.dv_data, self.dhw_data, self.thw_data])
        std_speed, std_gap, std_dv, std_dhw, std_thw = \
            (np.std(data) for data in [self.speed_data, self.gap_data, self.dv_data, self.dhw_data, self.thw_data])
        avg_q = 3600 / avg_thw
        avg_k = 1000 / avg_dhw
        q_divide_k = avg_q / avg_k / 3.6
        return {"avg_speed": avg_speed, "avg_gap": avg_gap, "avg_dv": avg_dv, "avg_dhw": avg_dhw, "avg_thw": avg_thw,
                "std_speed": std_speed, "std_gap": std_gap, "std_dv": std_dv, "std_dhw": std_dhw, "std_thw": std_thw,
                "avg_q": avg_q, "avg_k": avg_k, "q_divide_k": q_divide_k}

    def data_to_df(self):
        """环形边界一个车辆轨迹拆分为多段，id加后缀_x"""
        dict_ = {"Frame_ID": [], "v_ID": [], "Local_xVelocity": [], "Preceding_ID": [], "v_Length": [],
                 "Local_X": [], "gap": [], "dhw": [], "thw": [], "Local_xAcc": []}
        data_len = int(self.pos_data.shape[0])
        for i in range(self.car_num):
            for key in dict_.keys():
                temp: Optional[Iterable, object] = None
                if key == "Frame_ID":
                    temp = np.arange(self.warm_up_step + 1, self.sim_step).tolist()
                elif key == "v_ID":
                    count = 0
                    dict_["v_ID"].extend([i] * data_len)
                    dict_["Preceding_ID"].extend([(i + 1) if (i + 1 != self.car_num) else 0] * data_len)
                    for _, temp_, _, _ in self._data_shear(self.pos_data, index=i):
                        dict_["Local_X"].extend(temp_ + count * self.lane_length)
                        count += 1
                    continue
                elif key == "Local_xVelocity":
                    temp = self.speed_data[:, i]
                elif key == "Local_xAcc":
                    temp = self.acc_data[:, i]
                elif key == "v_Length":
                    temp = [self.car_length] * data_len
                elif key == "gap":
                    temp = self.gap_data[:, i]
                elif key == "dhw":
                    temp = self.dhw_data[:, i]
                elif key == "thw":
                    temp = self.thw_data[:, i]
                if temp is not None:
                    dict_[key].extend(temp)
        df = pd.DataFrame(dict_)
        df.to_csv("test.csv")

    def ui_init(self, caption="环形单车道车队跟驰"):
        pg.init()
        # self.screen_height =
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        pg.display.set_caption(caption)
        self.clock = pg.time.Clock()

        self.ui_update()

    def ui_update(self):
        self.screen.fill((0, 0, 0))
        for i in range(self.car_pos.shape[1]):
            # TODO: 实现转折或缩放
            # pos_y = int(self.car_pos[0, i] / 1000)
            # pos_x = self.car_pos[0, i] - pos_y * 1000
            pg.draw.rect(self.screen, (255, 0, 0),
                         (self.car_pos[0, i] / self.lane_length * self.screen_width, int(self.screen_height / 2),
                          int(self.car_length / self.lane_length * self.screen_width), 10))

        pg.display.update()
        if self.frame_rate > 0:
            self.clock.tick(self.frame_rate)

    def plot(self, index=0):
        """绘制车辆index"""
        time_ = np.arange(self.warm_up_step + 1, self.sim_step) * self.dt

        fig, axes = plt.subplots(2, 2, figsize=(7, 5), layout="constrained")
        axes: np.ndarray[plt.Axes] = axes

        ax = axes[0, 0]
        ax.set_xlabel("time")
        ax.set_ylabel("speed")
        ax.plot(time_, self.speed_data[:, index], label=f"index={index}")

        ax = axes[0, 1]
        ax.set_xlabel("speed")
        ax.set_ylabel("gap")
        ax.plot(self.speed_data[:, index], self.gap_data[:, index], label=f"index={index}")

        ax = axes[1, 0]
        ax.set_xlabel("dv")
        ax.set_ylabel("gap")
        ax.plot(self.dv_data[:, index], self.gap_data[:, index], label=f"index={index}")

        ax = axes[1, 1]
        ax.set_xlabel("time")
        ax.set_ylabel("acc")
        ax.plot(time_, self.acc_data[:, index], label=f"index={index}")

        fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout="constrained")
        ax: plt.Axes = ax
        fig: plt.Figure = fig
        ax.set_xlabel("time")
        ax.set_ylabel("location")
        lc = None
        value_range = (np.min(self.speed_data), np.max(self.speed_data))
        cmap = plt.get_cmap('rainbow')
        for time__, temp__, index_, pos in self._data_shear(self.pos_data):
            if pos[1] - pos[0] > 1:
                lc = self._lines_color(ax, time__, temp__, self.speed_data[:, index_][pos[0]: pos[1]], value_range,
                                       cmap, line_width=0.2 if index != index_ else 1)

        fig.colorbar(ScalarMappable(mc.Normalize(vmin=value_range[0], vmax=value_range[1]), cmap))

        plt.show()

    @staticmethod
    def _lines_color(ax, data_x, data_y, color_value, value_range, cmap, line_width=0.2):
        data_x = np.array(data_x).reshape(-1)
        data_y = np.array(data_y).reshape(-1)
        points = [(x, y) for x, y in zip(data_x, data_y)]
        color_value = np.array(color_value).reshape(-1)
        color_value = color_value[:-1]
        seg = np.array([(a, b) for a, b in zip(points[:-1], points[1:])])

        color_value = (color_value - value_range[0]) / (value_range[1] - value_range[0])
        colors = cmap(color_value)
        lc = mcoll.LineCollection(seg, colors=colors, linewidths=line_width)

        ax.add_collection(lc)
        ax.autoscale(True)
        return lc

    def _data_shear(self, data, index=-1):
        time_ = np.arange(self.warm_up_step + 1, self.sim_step) * self.dt

        for j in range(self.car_num):
            if index >= 0 and index != j:
                continue
            temp_ = data[:, j]
            return_index = list(np.where(np.diff(temp_) < 0)[0])
            return_index.insert(0, 0)
            for i in range(len(return_index)):
                if i == 0:
                    pos = (0, return_index[i + 1] + 1)
                elif i != len(return_index) - 1 and i != 0:
                    pos = (return_index[i] + 1, return_index[i + 1] + 1)
                else:
                    pos = (return_index[i] + 1, len(temp_))
                temp__ = temp_[pos[0]: pos[1]]
                time__ = time_[pos[0]: pos[1]]
                yield time__, temp__, j, pos

    def __str__(self):
        return "lane_length: " + str(self.lane_length) + \
            "\tcar_num: " + str(self.car_num) + \
            "\tcar_length: " + str(self.car_length) + \
            "\tcar_initial_speed: " + str(self.car_initial_speed) + \
            "\tcf_mode: " + self.cf_mode + \
            "\tcf_param: " + self.cf_model.get_param_map().__str__() + \
            "\tbasic_save: " + str(self.basic_save) + \
            "\taggregate_cal: " + str(self.aggregate_cal) + \
            "\tplot_data: " + str(self.plot_data) + \
            "\tui: " + str(self.ui) + \
            "\tframe_rate" + str(self.frame_rate) + \
            "\tdf_save: " + str(self.df_save) + \
            "\tdt: " + str(self.dt) + \
            "\twarm_up_step: " + str(self.warm_up_step) + \
            "\tsim_step: " + str(self.sim_step)
