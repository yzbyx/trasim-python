# -*- coding: utf-8 -*-
# @Time : 2023/11/18 22:23
# @Author : yzbyx
# @File : slow_to_go_hys.py
# Software: PyCharm
import numpy as np

from traj_process.analyzer.safety_analysis import TTC
from trasim_simplified.core.kinematics.cfm.CFModel_IDM import cf_IDM_acc_module, cf_IDM_equilibrium_module
from trasim_simplified.util.hysteresis.intensity import cal_project_to_x_axis_area, cal_sv_intensity
from trasim_simplified.util.hysteresis.sim_scenario import slow_to_go_sim
from trasim_simplified.util.interaction.quiver import QuiverInteract


class SlowToGo_Interact(QuiverInteract):
    def __init__(self, cf_func, cf_e_func, **kwargs):
        super().__init__(cf_func, cf_e_func, **kwargs)
        self.speed = 15
        self.interval = 0.1
        self.dv = 10

    def update(self, val):
        [ax.cla() for ax in self.axes]
        self._set_speed_range(self.get_params().get("speed", 10), self.get_params().get("dv", 3))
        self.draw_equilibrium()  # 得到图幅的边界
        self.draw_hysteresis()
        self.fig.canvas.draw()
        self.axes[0].set_title(self.cf_func.__name__)
        self.fig.canvas.draw_idle()

    def draw_hysteresis(self):
        dec_s, dec_v, acc_s, acc_v, dec_a, acc_a, dec_lv, acc_lv, dec_lx, acc_lx \
            = slow_to_go_sim(self.cf_func, self.get_cf_params(), cf_e=self.cf_e_func,
                             warmup_time=100, dec_time=10, slow_time=100,
                             acc_time=10, hold_time=100,
                             init_v=self.speed, dv=self.dv, dt=self.interval)
        vs_results = cal_sv_intensity(dec_s, dec_v, acc_s, acc_v, self.cf_e_func, self.get_cf_params())

        acc_vs = vs_results["acc_vs"]
        dec_vs = vs_results["dec_vs"]
        total_vs = vs_results["acc_vs"] + vs_results["dec_vs"]
        self.axes[0].text(acc_v[0], acc_s[0], f"acc_vs: {vs_results['acc_vs']:.2f}")
        self.axes[0].text(dec_v[0], dec_s[0], f"dec_vs: {vs_results['dec_vs']:.2f}")
        self.axes[0].text(acc_v[-1], acc_s[-1] + 2, f"total_vs: {total_vs:.2f}")

        self.axes[0].plot(dec_v, dec_s, "-o", markersize=1, color="blue")
        self.axes[0].plot(acc_v, acc_s, "-o", markersize=1, color="red")

        self.axes[0].scatter([dec_v[100]], [dec_s[100]], marker="<", c="r")
        self.axes[0].scatter([acc_v[100]], [acc_s[100]], marker=">", c="r")

        self.axes[1].plot(dec_a, "-o", markersize=1, color="blue", label="dec")
        self.axes[1].plot(acc_a, "-o", markersize=1, color="red", label="acc")
        self.axes[1].set_title("acceleration")
        self.axes[1].legend()

        self.axes[2].plot(dec_v, "-o", markersize=1, color="blue", label="dec")
        self.axes[2].plot(acc_v, "-o", markersize=1, color="red", label="acc")
        self.axes[2].set_title("velocity")
        self.axes[2].legend()

        dec_q = 1 / (dec_s / dec_v)
        acc_q = 1 / (acc_s / acc_v)
        self.axes[3].plot(dec_v, dec_q, "-o", markersize=1, color="blue", label="dec")
        self.axes[3].plot(acc_v, acc_q, "-o", markersize=1, color="red", label="acc")
        self.axes[3].set_title("flow")
        self.axes[3].legend()

        dec_ttc = TTC(dec_v - dec_lv, dec_s, cutoff=100)
        acc_ttc = TTC(acc_v - acc_lv, acc_s, cutoff=100)
        self.axes[4].plot(dec_ttc, "-o", markersize=1, color="blue", label="dec")
        self.axes[4].plot(acc_ttc, "-o", markersize=1, color="red", label="acc")
        self.axes[4].set_title("TTC")
        self.axes[4].legend()


if __name__ == '__main__':
    SlowToGo_Interact(
        cf_func=cf_IDM_acc_module,
        cf_e_func=cf_IDM_equilibrium_module,
        default={"s0": 2, "s1": 0, "v0": 33.3, "T": 1.6, "omega": 0.73, "d": 1.67, "delta": 4,
                 "k_speed": 1, "k_space": 1, "k_zero": 1, "speed": 10, "dv": 3},
        range={"s0": [0, 5], "s1": [0, 5], "v0": [0, 40], "T": [0, 5], "omega": [0, 5], "d": [0, 5], "delta": [0, 10],
               "k_speed": [0, 10], "k_space": [0, 10], "k_zero": [0, 10], "speed": [5, 40], "dv": [0, 20]},
        step={"s0": 0.1, "s1": 0.1, "v0": 0.1, "T": 0.1, "omega": 0.1, "d": 0.1, "delta": 1,
              "k_speed": 0.1, "k_space": 0.1, "k_zero": 0.1, "speed": 0.1, "dv": 0.1}
    ).run()
