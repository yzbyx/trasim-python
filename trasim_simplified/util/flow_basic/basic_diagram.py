# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:43
# @Author : yzbyx
# @File : basic_diagram.py
# @Software : PyCharm
import os
import pickle
import time
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from trasim_simplified.core.constant import CFM
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.circle_frame import FrameCircle
from trasim_simplified.core.data.data_processor import Info as P_Info
from trasim_simplified.core.data.data_container import Info as C_Info


class BasicDiagram:
    def __init__(self, lane_length: int, car_length: int, car_initial_speed:int, speed_with_random: bool,
                 cf_mode: str, cf_param: dict[str, float]):
        self.car_length = car_length
        self.lane_length = lane_length
        self.car_initial_speed = car_initial_speed
        self.speed_with_random = speed_with_random
        self.cf_mode = cf_mode
        self.cf_param = cf_param

        self.resume = False

        self.occ_seq = None
        self.result: Optional[dict[str, list]] = {"occ": [], "Q": [], "K": [], "V": []}

    def run(self, occ_start: float, occ_end: float, d_occ: float, resume=False):
        self.resume = resume
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

            if self.check_contain_occ(self.occ_seq[i]): continue

            print(f"[{str(i + 1).zfill(3)}/{str(len(car_nums)).zfill(3)}]"
                  f" occ: {occ_seq[i]:.2f}, car_nums: {round(car_num)}", end="\t\t\t")

            frame = FrameCircle(self.lane_length, car_num, self.car_length, self.car_initial_speed,
                                self.speed_with_random, self.cf_mode, self.cf_param)
            frame.data_container.config(save_info={C_Info.v})
            for _ in frame.run(data_save=True, has_ui=False, warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
                pass
            result = frame.data_processor.kqv_cal()
            self.result["occ"].append(occ_seq[i])
            self.result["V"].append(np.mean(result[P_Info.avg_speed]) * 3.6)
            self.result["Q"].append(np.mean(result[P_Info.avg_q_by_v_k]))
            self.result["K"].append(np.mean(result[P_Info.avg_k_by_car_num_lane_length]))

            self.save_result()

            time_epoch_end = time.time()
            time_epoch = time_epoch_end - time_epoch_begin
            cal_speed = car_num * sim_step / time_epoch
            time_ += time_epoch
            print(f"time_used: {time_:.2f}s + time_epoch: {time_epoch:.2f}s + cal_speed: {cal_speed:.2f}cal/s^-1")

    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(10, 2), layout="constrained", squeeze=False)
        axes: list[list[plt.Axes]] = axes

        ax = axes[0][0]
        Plot.custom_plot(ax, "Occ", "Q(veh/h)", [self.occ_seq], [self.result["Q"]], data_label="Q-Occ")

        ax = axes[0][1]
        Plot.custom_plot(ax, "Q(veh/h)", "V(km/h)", [self.result["Q"]], [self.result["V"]], data_label="V-Q")

        ax = axes[0][2]
        Plot.custom_plot(ax, "K(veh/km)", "V(km/h)", [self.result["K"]], [self.result["V"]], data_label="V-K")

        plt.show()

    def save_result(self):
        with open("./temp/result.pkl", "wb") as f:
            pickle.dump(self.result, f)

    def load_result(self):
        if not self.resume:
            self.clear_result()
            return {"occ": [], "Q": [], "K": [], "V": []}
        else:
            with open("./temp/result.pkl", "rb") as f:
                return pickle.load(f)

    @staticmethod
    def clear_result():
        os.remove("./temp/result.pkl")

    def check_contain_occ(self, occ):
        if not self.resume: return False
        for occ_ in self.result["occ"]:
            if abs(occ_ - occ) < 1e-6:
                return True
        return False


if __name__ == '__main__':
    diag = BasicDiagram(1000, 5, 0, False, cf_mode=CFM.IDM, cf_param={})
    diag.run(0.01, 0.7, 0.02)
    diag.plot()
