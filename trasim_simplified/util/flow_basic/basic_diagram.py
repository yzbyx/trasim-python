# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:43
# @Author : yzbyx
# @File : basic_diagram.py
# @Software : PyCharm
import time
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from trasim_simplified.core.constant import CFM
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.circle_frame import FrameCircle


class BasicDiagram:
    def __init__(self, lane_length: int, car_length: int, car_initial_speed:int, speed_with_random: bool,
                 cf_mode: str, cf_param: dict[str, float]):
        self.car_length = car_length
        self.lane_length = lane_length
        self.car_initial_speed = car_initial_speed
        self.speed_with_random = speed_with_random
        self.cf_mode = cf_mode
        self.cf_param = cf_param

        self.occ_seq = None
        self.result: Optional[dict[str, list]] = {"Q": [], "K": [], "V": []}

    def run(self, occ_start: float, occ_end: float, d_occ: float,):
        assert 0 < occ_start <= (occ_end - d_occ) <= 1, "请输入有效的occ起始点！"
        assert d_occ * self.lane_length >= self.car_length,\
            f"每次occ的增加量至少能够增加一辆车，最小occ步长为{d_occ * self.lane_length}"
        if occ_end - occ_start <= d_occ:
            occ_seq = np.array([occ_start])
        else:
            occ_seq = np.round(np.arange(occ_start, occ_end, d_occ), 6)
        car_nums = np.round(np.ceil(occ_seq * self.lane_length / self.car_length))
        self.occ_seq = occ_seq
        warm_up_step = 6000 * 2
        sim_step = 6000 * 3
        dt = 0.1

        time_ = 0
        for i, car_num in enumerate(car_nums):
            time_epoch_begin = time.time()
            print(f"[{str(i + 1).zfill(3)}/{str(len(car_nums)).zfill(3)}]"
                  f" occ: {occ_seq[i]:.2f}, car_nums: {round(car_num)}", end="\t\t\t")

            frame = FrameCircle(self.lane_length, car_num, self.car_length, self.car_initial_speed,
                                self.speed_with_random, self.cf_mode, self.cf_param)
            for _ in frame.run(basic_save=True, has_ui=False, warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
                pass
            result = frame.data_processor.aggregate()
            self.result["V"].append(np.mean(result["avg_speed(m/s)"]) * 3.6)
            self.result["Q"].append(np.mean(result["avg_q(v*k)(veh/h)"]))
            self.result["K"].append(np.mean(result["avg_k(veh/km)"]))

            time_epoch_end = time.time()
            time_epoch = time_epoch_end - time_epoch_begin
            cal_speed = car_num * sim_step / time_epoch
            time_ += time_epoch
            print(f"time_used: {time_:.2f}s + time_epoch: {time_epoch:.2f}s + cal_speed: {cal_speed:.2f}cal/s^-1")

    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5), layout="constrained", squeeze=False)
        axes: list[list[plt.Axes]] = axes

        ax = axes[0][0]
        Plot.custom_plot(ax, "Occ", "Q(veh/h)", self.occ_seq, self.result["Q"], data_label="Q-Occ")

        ax = axes[0][1]
        Plot.custom_plot(ax, "Q(veh/h)", "V(km/h)", self.result["Q"], self.result["V"], data_label="V-Q")

        ax = axes[0][2]
        Plot.custom_plot(ax, "K(veh/km)", "V(km/h)", self.result["K"], self.result["V"], data_label="V-K")


if __name__ == '__main__':
    diag = BasicDiagram(1000, 5, 0, False, cf_mode=CFM.IDM, cf_param={})
    diag.run(0.1, 1.01, 0.1)
    diag.plot()
