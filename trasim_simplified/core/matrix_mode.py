# -*- coding = uft-8 -*-
# @Time : 2023-03-24 16:21
# @Author : yzbyx
# @File : matrix_mode.py
# @Software : PyCharm
from timeit import timeit
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame as pg
from pygame.time import Clock

from trasim_simplified.core.constant import CFM
from trasim_simplified.kinematics.cfm import get_cf_model
from trasim_simplified.util.decorator.mydecorator import timer_no_log


class MatrixMode:
    def __init__(self, lane_length: int, car_num: int, car_length: int, car_initial_speed:int,
                 cf_mode: str, cf_param: dict[str, float]):
        dhw = lane_length / car_num
        assert dhw >= car_length, f"该密度下，车辆重叠！此车身长度下车辆数最多为{np.floor(lane_length / car_length)}"
        self.car_num = car_num
        self.car_length = car_length
        self.lane_length = lane_length
        self.car_pos = np.arange(0, lane_length, dhw).reshape(1, -1)
        assert car_num == self.car_pos.shape[1], f"车辆生成数量有误！目标：{car_num}，结果：{self.car_pos.shape}"
        self.car_speed = np.random.uniform(
            max(car_initial_speed - 0.5, 0),  car_initial_speed + 0.5, self.car_pos.shape
        ).reshape(1, -1)
        self.car_acc = np.zeros(self.car_pos.shape).reshape(1, -1)

        self.cf_model = get_cf_model(None, cf_mode, cf_param)
        self.cf_param = self.cf_model.get_param_map()

        self.step_ = 0
        """当前仿真时间 [s]"""

        self.dt = 0.1
        """仿真步长 [s]"""
        self.warm_up_step = int(5 * 60 / self.dt)
        """预热时间 [s]"""
        self.sim_step = int(10 * 60 / self.dt)
        """总仿真步 [次]"""

        self.speed_data: Optional[np.ndarray] = None
        self.pos_data: Optional[np.ndarray] = None
        self.gap_data: Optional[np.ndarray] = None
        self.dhw_data: Optional[np.ndarray] = None
        self.thw_data: Optional[np.ndarray] = None
        self.dv_data: Optional[np.ndarray] = None

        self.basic_save = False
        self.aggregate_cal = False
        self.plot_data = False
        self.dpi = 500

        self.ui = False
        self.screen_width = 1980 / 1.5
        self.screen_height = 100 / 1.5
        self.screen: Optional[pg.Surface] = None
        self.clock: Optional[Clock] = None

    @timer_no_log
    def run(self, basic_save=True, aggregate_cal=True, plot_data=True, df_save=True, ui=True, **kwargs):
        self.basic_save = basic_save
        self.aggregate_cal = aggregate_cal
        self.plot_data = plot_data
        self.ui = ui

        if kwargs is None:
            kwargs = {}
        self.dt = kwargs.get("dt", 0.1)
        """仿真步长 [s]"""
        self.warm_up_step = kwargs.get("warm_up_step", int(5 * 60 / self.dt))
        """预热步数 [s]"""
        self.sim_step = kwargs.get("sim_step", int(10 * 60 / self.dt))
        """总仿真步 [次]"""

        if ui: self.ui_init()

        while self.sim_step != self.step_:
            self.step()
            if self.ui: self.ui_update()

        if aggregate_cal and basic_save:
            print(self.aggregate())
            if plot_data: self.plot()
            if df_save: self.data_to_df()

    def data_to_df(self):
        dict_ = {"Frame_ID": [], "v_ID": [], "Local_xVelocity": [], "Local_X": [], "gap": [], "dhw": [], "thw": [] }
        # df = pd.DataFrame({"Frame_ID": range(self.speed_data.shape[]), })

    def ui_init(self):
        pg.init()
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        pg.display.set_caption("环形单车道车队跟驰")
        self.clock = pg.time.Clock()

        self.ui_update()

    def ui_update(self):
        self.screen.fill((0, 0, 0))
        for i in range(self.car_pos.shape[1]):
            pg.draw.rect(self.screen, (255, 0, 0),
                         (self.car_pos[0, i] / self.lane_length * self.screen_width, int(self.screen_height / 2),
                         int(self.car_length / self.lane_length * self.screen_width), 20))
        # 刷新屏幕
        pg.display.update()
        self.clock.tick(240)

    def step(self):
        if self.basic_save and self.step_ > self.warm_up_step:
            self.record()

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
        car_speed_before = self.car_speed
        self.car_speed += self.car_acc * self.dt
        self.car_pos += (car_speed_before + self.car_speed) / 2 * self.dt
        self.car_pos[np.where(self.car_pos > self.lane_length)] -= self.lane_length

        self.step_ += 1

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
        self.speed_data = np.concatenate([self.speed_data, self.car_speed], axis=0) \
            if self.speed_data is not None else self.car_speed
        current_dhw = self._get_dhw(self.car_pos)
        self.dhw_data = np.concatenate([self.dhw_data, current_dhw], axis=0)\
            if self.dhw_data is not None else current_dhw
        self.gap_data = np.concatenate([self.gap_data, current_dhw - self.car_length], axis=0)\
            if self.gap_data is not None else current_dhw - self.car_length
        self.thw_data = np.concatenate([self.thw_data, current_dhw / self.car_speed], axis=0) \
            if self.thw_data is not None else current_dhw / self.car_speed
        self.dv_data = np.concatenate([self.dv_data, self._get_dv(self.car_speed)], axis=0) \
            if self.dv_data is not None else self._get_dv(self.car_speed)
        self.pos_data = np.concatenate([self.pos_data, self.car_pos], axis=0) \
            if self.pos_data is not None else self.car_pos

    def aggregate(self):
        assert self.speed_data.size != 0 and self.gap_data.size != 0 and self.dhw_data.size != 0 and \
               self.thw_data.size != 0 and self.dv_data.size != 0 and self.pos_data.size != 0,\
            "调用本函数须使用record函数记录数据"
        avg_speed, avg_gap, avg_dv, avg_dhw, avg_thw =\
            (np.mean(data) for data in [self.speed_data, self.gap_data, self.dv_data, self.dhw_data, self.thw_data])
        std_speed, std_gap, std_dv, std_dhw, std_thw =\
            (np.std(data) for data in [self.speed_data, self.gap_data, self.dv_data, self.dhw_data, self.thw_data])
        avg_q = 3600 / avg_thw
        avg_k = 1000 / avg_dhw
        q_divide_k = avg_q / avg_k / 3.6
        return {"avg_speed": avg_speed, "avg_gap": avg_gap, "avg_dv": avg_dv, "avg_dhw": avg_dhw, "avg_thw": avg_thw,
                "std_speed": std_speed, "std_gap": std_gap, "std_dv": std_dv, "std_dhw": std_dhw, "std_thw": std_thw,
                "avg_q": avg_q, "avg_k": avg_k, "q_divide_k": q_divide_k}

    def plot(self, index=0):
        """绘制车辆index"""
        time_ = np.arange(self.warm_up_step + 1, self.sim_step) * self.dt

        fig, axes = plt.subplots(2, 2, figsize=(7, 5), layout="constrained")
        axes: np.ndarray[plt.Axes] = axes

        ax = axes[0, 0]
        ax.set_xlabel("time")
        ax.set_ylabel("speed")
        ax.plot(time_, self.speed_data[:, index])

        ax = axes[0, 1]
        ax.set_xlabel("speed")
        ax.set_ylabel("gap")
        ax.plot(self.speed_data[:, index], self.gap_data[:, index])

        ax = axes[1, 0]
        ax.set_xlabel("dv")
        ax.set_ylabel("gap")
        ax.plot(self.dv_data[:, index], self.gap_data[:, index])

        ax = axes[1, 1]
        ax.set_xlabel("time")
        ax.set_ylabel("thw")
        ax.plot(time_, self.thw_data[:, index])

        fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout="constrained")
        for j in range(self.car_num):
            temp_ = self.pos_data[:, j]
            return_index = list(np.where(np.diff(temp_) < 0)[0])
            return_index.insert(0, 0)
            for i in range(len(return_index)):
                if i == 0:
                    temp__ = temp_[: return_index[i + 1] + 1]
                    time__ = time_[: return_index[i + 1] + 1]
                elif i != len(return_index) - 1 and i != 0:
                    temp__ = temp_[return_index[i] + 1: return_index[i + 1] + 1]
                    time__ = time_[return_index[i] + 1 : return_index[i + 1] + 1]
                else:
                    temp__ = temp_[return_index[i] + 1:]
                    time__ = time_[return_index[i] + 1:]
                ax.plot(time__, temp__, linewidth=0.2, color='black')

        plt.show()


def run():
    _cf_param = {}
    sim = MatrixMode(1000, 60, 5, 0, CFM.IDM, _cf_param)
    sim.run(basic_save=True, ui=False, df_save=True, warm_up_step=3000, sim_step=3600, dt=1)


if __name__ == '__main__':
    run()
