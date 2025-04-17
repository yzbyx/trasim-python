# -*- coding = uft-8 -*-
# @time : 2023-03-31 16:43
# @Author : yzbyx
# @File : basic_diagram.py
# @Software : PyCharm
import os
import pickle
import threading
import time
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from trasim_simplified.core.constant import V_TYPE
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.micro.circle_lane import LaneCircle
from trasim_simplified.core.data.data_processor import Info as P_Info
from trasim_simplified.core.data.data_container import Info as C_Info
from trasim_simplified.core.kinematics.cfm import get_cf_model


class BasicDiagram:
    def __init__(self, lane_length: int, car_length: float, car_initial_speed: float, speed_with_random: bool,
                 cf_mode: str, cf_param: dict[str, float], speed_limit=30):
        self.car_nums = None
        self.car_length = car_length
        self.lane_length = lane_length
        self.speed_limit = speed_limit
        self.car_initial_speed = car_initial_speed
        self.speed_with_random = speed_with_random
        self.cf_mode = cf_mode
        self.cf_param = cf_param

        self.resume = False
        self.file_name = None

        self.occ_seq = None
        self.result: Optional[dict[str, list]] = {"occ": [], "Q": [], "K": [], "V": []}
        self.equilibrium_state_result = {"Q": [], "K": [], "V": []}

    def run(self, occ_start: float, occ_end: float, d_occ: float, resume=False, file_name=None, **kwargs):
        self.resume = resume
        assert 0 < occ_start <= (occ_end - d_occ) <= 1, "请输入有效的occ起始点！"
        assert d_occ * self.lane_length >= self.car_length, \
            f"每次occ的增加量至少能够增加一辆车，最小occ步长为{d_occ * self.lane_length}"
        if occ_end - occ_start <= d_occ:
            occ_seq = np.array([occ_start])
        else:
            occ_seq = np.round(np.arange(occ_start, occ_end, d_occ), 6)
        self.car_nums = list(map(int, np.round(np.ceil(occ_seq * self.lane_length / self.car_length))))
        self.occ_seq = np.round(np.array(self.car_nums) * self.car_length / self.lane_length, 6)

        self.file_name = file_name if file_name is not None else "result_IDM.pkl"
        self.result = self.load_result(self.resume, self.file_name)

        dt = kwargs.get("dt", 0.1)
        warm_up_step = int(300 / dt)
        sim_step = warm_up_step + int(300 / dt)
        jam = kwargs.get("jam", False)
        update_method = kwargs.get("state_update_method", "Euler")

        print(f"CFM: {self.cf_mode}, is_jam: {jam}")

        time_ = time.time()

        def cal(i, car_num):
            time_epoch_begin = time.time()

            if not kwargs.get("parallel", False) and self.check_contain_occ(self.occ_seq[i]):
                return

            params = [car_num, self.car_length, V_TYPE.PASSENGER, self.car_initial_speed,
                      self.speed_with_random, self.cf_mode, self.cf_param, {}]

            lane = LaneCircle(self.lane_length)
            lane.set_speed_limit(self.speed_limit)
            lane.car_config(*params)
            gap = 0 if jam else -1
            lane.car_load(gap)

            lane.data_container.add_basic_info()
            lane.data_container.config(save_info={C_Info.v})

            for _ in lane.run(data_save=True, has_ui=False, warm_up_step=warm_up_step, sim_step=sim_step, dt=dt,
                              state_update_method=update_method, force_speed_limit=False):
                pass

            df = lane.data_container.data_to_df()

            result = lane.data_processor.circle_kqv_cal(df, self.lane_length)
            if not kwargs.get("parallel", False):
                self.result["occ"].append(self.occ_seq[i])
                self.result["V"].append(np.mean(result[P_Info.avg_speed]) * 3.6)
                self.result["Q"].append(np.mean(result[P_Info.avg_q_by_v_k]))
                self.result["K"].append(np.mean(result[P_Info.avg_k_by_car_num_lane_length]))

                self.save_result(self.file_name, self.result)

            time_epoch_end = time.time()
            time_epoch = time_epoch_end - time_epoch_begin
            cal_speed = car_num * sim_step / time_epoch
            print(
                f"Thread: {threading.current_thread().native_id} "
                f"[{str(i + 1).zfill(3)}/{str(len(self.car_nums)).zfill(3)}]"
                f" occ: {self.occ_seq[i]:.2f}, car_nums: {car_num},"
                f" density[veh/km]: {car_num / (self.lane_length / 1000)}",
                f"\t\t\t"
                f"time_used: {time.time() - time_:.2f}s "
                f"time_epoch: {time_epoch:.2f}s "
                f"cal_speed: {cal_speed:.2f}cal/s^-1"
            )
            if kwargs.get("parallel", True):
                return self.occ_seq[i], np.mean(result[P_Info.avg_speed]) * 3.6, np.mean(result[P_Info.avg_q_by_v_k]), \
                          np.mean(result[P_Info.avg_k_by_car_num_lane_length])

        if not kwargs.get("parallel", False):
            for i_, car_num_ in enumerate(self.car_nums):
                cal(i_, car_num_)
        else:
            self.result["occ"], self.result["V"], self.result["Q"], self.result["K"] \
                = zip(*(Parallel(n_jobs=-1)(delayed(cal)(i, car_num) for i, car_num in enumerate(self.car_nums))))
            self.save_result(self.file_name, self.result)

    def get_by_equilibrium_state_func(self):
        cf_model = get_cf_model(None, self.cf_mode, self.cf_param)
        for i, speed in enumerate(self.result["V"]):
            car_num = (self.result["occ"][i] * self.lane_length) / self.car_length
            dhw = self.lane_length / car_num
            speed /= 3.6
            result = cf_model.equilibrium_state(speed, dhw, self.car_length)
            if result is not None:
                self.equilibrium_state_result["V"].append(result["V"] * 3.6)
                self.equilibrium_state_result["Q"].append(result["Q"] * 3600)
                self.equilibrium_state_result["K"].append(result["K"] * 1000)

    def plot(self, save_fig=True):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained", squeeze=False)
        fig: plt.Figure = fig
        axes: list[list[plt.Axes]] = axes

        ax = axes[0][0]
        Plot.custom_plot(ax, "K(veh/km)", "Q(veh/h)",
                         [self.result["K"]],
                         [self.result["Q"]], data_label="Q-Occ",
                         color='blue', marker='s', linestyle='solid',
                         linewidth=1, markersize=2)
        if len(self.equilibrium_state_result["V"]) != 0:
            Plot.custom_plot(ax, "K(veh/km)", "Q(veh/h)", [self.result["K"]], [self.equilibrium_state_result["Q"]],
                             data_label="Q-Occ-E", color='green', marker='s', linestyle='dashed',
                             linewidth=1, markersize=2)

        ax = axes[0][1]
        Plot.custom_plot(ax, "Q(veh/h)", "V(km/h)", [self.result["Q"]], [self.result["V"]], data_label="V-Q",
                         color='blue', marker='s', linestyle='solid',
                         linewidth=1, markersize=2)
        if len(self.equilibrium_state_result["V"]) != 0:
            Plot.custom_plot(ax, "Q(veh/h)", "V(km/h)", [self.equilibrium_state_result["Q"]],
                             [self.equilibrium_state_result["V"]], data_label="V-Q-E",
                             color='green', marker='s', linestyle='dashed',
                             linewidth=1, markersize=2)

        ax = axes[0][2]
        Plot.custom_plot(ax, "K(veh/km)", "V(km/h)", [self.result["K"]], [self.result["V"]], data_label="V-K",
                         color='blue', marker='s', linestyle='solid',
                         linewidth=1, markersize=2)
        if len(self.equilibrium_state_result["V"]) != 0:
            Plot.custom_plot(ax, "K(veh/km)", "V(km/h)", [self.equilibrium_state_result["K"]],
                             [self.equilibrium_state_result["V"]], data_label="V-K-E",
                             color='green', marker='s', linestyle='dashed',
                             linewidth=1, markersize=2)

        fig.suptitle(self.cf_mode + "+" + get_cf_model(None, self.cf_mode, self.cf_param).get_param_map().__str__())
        if save_fig: fig.savefig("./diag_result/" + self.file_name + ".png", dpi=500, bbox_inches='tight')
        plt.show()

    @staticmethod
    def save_result(file_name, result, full_path=False):
        if not full_path:
            with open(f"./diag_result/{file_name}.pkl", "wb") as f:
                pickle.dump(result, f)
        else:
            with open(f"{file_name}", "wb") as f:
                pickle.dump(result, f)

    @staticmethod
    def load_result(resume, file_name, full_path=False):
        if not resume:
            BasicDiagram.clear_result(file_name, full_path)
            return {"occ": [], "Q": [], "K": [], "V": []}
        else:
            if not full_path:
                with open(f"./diag_result/{file_name}.pkl", "rb") as f:
                    return pickle.load(f)
            else:
                with open(f"{file_name}", "rb") as f:
                    return pickle.load(f)

    @staticmethod
    def clear_result(file_name, full_path=False):
        if not full_path:
            if os.path.exists(f"./diag_result/{file_name}.pkl"):
                os.remove(f"./diag_result/{file_name}.pkl")
        else:
            if os.path.exists(f"{file_name}"):
                os.remove(f"{file_name}")

    @staticmethod
    def overlay_map(file_name_list: list[str], labels=None, full_path=False):
        """将基本图进行叠加"""
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained", squeeze=False)
        fig: plt.Figure = fig
        axes: list[list[plt.Axes]] = axes

        result_list = []
        for file_name in file_name_list:
            result_list.append(BasicDiagram.load_result(True, file_name, full_path))

        if labels is None:
            labels = list(range(len(file_name_list)))

        ax = axes[0][0]
        Plot.custom_plot(ax, "Occ", "Q(veh/h)", [result["occ"] for result in result_list],
                         [result["Q"] for result in result_list], data_label=[str(label) for label in labels],
                         linewidth=2, markersize=3, marker="s")

        ax = axes[0][1]
        Plot.custom_plot(ax, "Q(veh/h)", "V(km/h)", [result["Q"] for result in result_list],
                         [result["V"] for result in result_list], data_label=[str(label) for label in labels],
                         linewidth=2, markersize=3, marker="s")

        ax = axes[0][2]
        Plot.custom_plot(ax, "K(veh/km)", "V(km/h)", [result["K"] for result in result_list],
                         [result["V"] for result in result_list], data_label=[str(label) for label in labels],
                         linewidth=2, markersize=3, marker="s")

        plt.show()

    def check_contain_occ(self, occ):
        if not self.resume: return False
        for occ_ in self.result["occ"]:
            if abs(occ_ - occ) < 1e-6:
                return True
        return False


if __name__ == '__main__':
    BasicDiagram.overlay_map([
        r"E:\PyProject\car-following-model-test\tests\diag_result\result_Three-Phase_TPACC_-1_False.pkl",
        r"E:\PyProject\car-following-model-test\tests\diag_result\result_Three-Phase_TPACC_0_False.pkl"
    ], labels=["maxV ", "still "], full_path=True)
