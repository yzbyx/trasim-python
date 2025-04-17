# -*- coding: utf-8 -*-
# @time : 2023/10/11 9:44
# @Author : yzbyx
# @File : quiver.py
# Software: PyCharm
import threading

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider

from trasim_simplified.core.kinematics.cfm.CFModel_IDM import cf_IDM_acc, cf_IDM_equilibrium
from trasim_simplified.util.hysteresis.intensity import cal_project_to_x_axis_area


class QuiverInteract:
    def __init__(self, cf_func, cf_e_func, fig_size=(14, 5), **kwargs):
        """
        **kwargs的结构：{"default": {"a": 0, ....}, "range: {"a": [0, 5], ...}", "step": {"a": 0.1, ...}}
        """
        self.cf_func = cf_func
        self.cf_e_func = cf_e_func

        self.slider_map = {}
        self.button = None
        self.defaults: dict = kwargs["default"]
        self.ranges: dict = kwargs["range"]
        self.steps: dict = kwargs["step"]

        self.interval = 0.1
        self.current_limit = None

        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        axes = axes.flatten()
        self.fig: plt.Figure = fig
        self.axes: list[plt.Axes] = axes
        self.info_ax: plt.Axes = self.fig.add_axes([0.7, 0.1, 0.2, 0.04])

    def _set_speed_range(self, speed, dv):
        self.speed = speed
        self.dv = dv
        self.speedRange = [self.speed - dv - 2, self.speed + dv + 2]

    def run(self):
        self.info_ax.set_axis_off()
        self.set_slider()
        self.set_button()
        self.update(0)
        plt.show()

    def set_slider(self):
        param_num = len(self.defaults)
        # adjust the main plot to make room for the sliders
        self.fig.subplots_adjust(right=0.5)
        bar_height_frac = 0.0225
        bar_length_frac = 0.4

        for i, (k, v) in enumerate(self.defaults.items()):
            ax_param = self.fig.add_axes([0.55, 0.2 + 0.8 / param_num * i, bar_length_frac, bar_height_frac])
            param_slider = Slider(
                ax=ax_param,
                label=k,
                valmin=self.ranges[k][0],
                valmax=self.ranges[k][1],
                valinit=v,
                valstep=self.steps[k]
            )
            param_slider.on_changed(self.update)
            self.slider_map.update({k: param_slider})

    def update(self, val):
        self.axes.cla()
        self._set_speed_range(self.get_params().get("speed", 10), self.get_params().get("dv", 3))
        self.draw_equilibrium()  # 得到图幅的边界
        t1 = threading.Thread(name="draw_quiver", target=self.draw_quiver)
        t2 = threading.Thread(name="draw_traj", target=self.draw_hysteresis)
        self.show_wait()
        self.fig.canvas.draw()
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.show_ok()
        # self.ax.set_ylim(self.gapRange.min(), self.gapRange.max())
        self.axes.set_title(self.cf_func.__name__)
        self.fig.canvas.draw_idle()

    def show_wait(self):
        self.info_ax.cla()
        self.info_ax.text(0, 0, "wait")
        self.info_ax.set_axis_off()

    def show_ok(self):
        self.info_ax.cla()
        self.info_ax.text(0, 0, "ok")
        self.info_ax.set_axis_off()

    def get_cf_params(self):
        params_dict = self.get_params()
        for k in ["speed", "dv"]:
            if k in params_dict:
                del params_dict[k]
        return params_dict

    def get_params(self):
        return {k: slider.val for k, slider in self.slider_map.items()}

    def draw_quiver(self):
        numX = round((self.speedRange[1] - self.speedRange[0]) / 0.3)
        vRange = np.linspace(self.speedRange[0], self.speedRange[1], num=numX)

        numY = round((self.current_limit[1][1] - self.current_limit[1][0]) / 0.3)
        gapRange = np.linspace(self.current_limit[1][1], self.current_limit[1][0], num=numY)

        dataX, dataY = np.meshgrid(vRange, gapRange)

        resultAcc = [[self.cf_func(speed=dataX[j][i], gap=dataY[j][i], leaderV=self.speed, interval=self.interval,
                                   **self.get_cf_params())
                      for i in range(numX)] for j in range(numY)]  # 存储顺序与书写顺序相同, 储存v的结果

        resultDeltaV = [[resultAcc[j][i] * self.interval for i in range(numX)] for j in range(numY)]

        resultDeltaGap = [
            [self.speed * self.interval - (dataX[j][i] * self.interval)
             for i in range(numX)] for j in range(numY)]

        resultDeltaV = np.array(resultDeltaV)
        resultDeltaGap = np.array(resultDeltaGap)
        resultAcc = np.array(resultAcc)

        self.axes.quiver(dataX, dataY, resultDeltaV, resultDeltaGap, np.abs(resultAcc), pivot="tail", scale=1,
                         units='inches', angles='xy', alpha=1, width=0.01, cmap=None, scale_units='xy')

    def draw_hysteresis(self):
        dec_speeds, dec_gaps = self.get_gap_v_data(self.speed + self.dv)
        dec_area = cal_project_to_x_axis_area(dec_speeds, dec_gaps)
        dec_e_speeds, dec_e_gaps = self.get_e_v_s(speedRange=[self.speed, self.speed + self.dv])
        e_area = cal_project_to_x_axis_area(dec_e_speeds, dec_e_gaps)
        dec_vs = (dec_area + e_area) / 2

        acc_speeds, acc_gaps = self.get_gap_v_data(self.speed - self.dv)
        acc_area = cal_project_to_x_axis_area(acc_speeds, acc_gaps)
        acc_e_speeds, acc_e_gaps = self.get_e_v_s(speedRange=[self.speed - self.dv, self.speed])
        e_area = cal_project_to_x_axis_area(acc_e_speeds, acc_e_gaps)
        acc_vs = (acc_area - e_area) / 2

        self.axes.text(acc_e_speeds[0], acc_e_gaps[0], f"acc_vs: {acc_vs:.2f}")
        self.axes.text(dec_e_speeds[-1], dec_e_gaps[-1], f"dec_vs: {dec_vs:.2f}")

        self.axes.plot(dec_speeds, dec_gaps)
        self.axes.plot(acc_speeds, acc_gaps)

    def get_gap_v_data(self, initial_v) -> tuple[list, list]:
        speed = initial_v
        params = self.get_cf_params()
        gap = self.cf_e_func(speed=initial_v, **params)
        acc = 0
        gaps = [gap]
        speeds = [speed]
        while len(speeds) < 20 or np.abs(acc) > 1e-5:
            acc = self.cf_func(speed=speed, leaderV=self.speed, gap=gap, **params)
            speed += acc * self.interval
            gap += self.speed * self.interval - speed * self.interval
            speeds.append(speed)
            gaps.append(gap)
        return speeds, gaps

    def draw_equilibrium(self):
        vRange, gaps = self.get_e_v_s()
        self.axes[0].plot(vRange, gaps)
        self.current_limit = (self.speedRange, (np.min(gaps), np.max(gaps)))

    def get_e_v_s(self, speedRange=None):
        if speedRange is None:
            speedRange = self.speedRange
        numX = round((speedRange[1] - speedRange[0]) / 0.1)
        vRange = np.linspace(speedRange[0], speedRange[1], num=numX)
        gaps = self.cf_e_func(speed=vRange, **self.get_cf_params())
        return vRange, gaps

    def set_button(self):
        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        reset_ax = self.fig.add_axes([0.55, 0.1, 0.1, 0.04])
        self.button = Button(reset_ax, 'Reset', hovercolor='0.975')

        self.button.on_clicked(self.reset)

    def reset(self, event):
        for _, slider in self.slider_map.items():
            slider.reset()


def cf_Zhang_acc(alpha, beta, v0, s0, T, speed, leaderV, gap, **kwargs):
    acc = alpha * (np.min(((gap - s0) / T, v0)) - speed) + beta * (leaderV - speed)
    return acc


def cf_Zhang_equilibrium(alpha, beta, v0, s0, T, v, **kwargs):
    if alpha == 0:
        return np.nan * v
    gap = v * T + s0
    return gap


def cf_OVM_acc(a, V0, m, bf, bc, speed, xOffset, leaderX, leaderL):
    bf += leaderL
    bc += leaderL  # 期望最小车头间距

    headway = leaderX - xOffset
    V = V0 * (np.tanh(m * (headway - bf)) - np.tanh(m * (bc - bf)))
    # V = V0 * (np.tanh(gap / b - C1) + C2)
    finalAcc = a * (V - speed)

    return finalAcc


def draw_quiver(ax: plt.Axes, speed_range, cf_func, cf_param, lv, dt, gap_range=None, cf_e=None, num_x=11, num_y=11):
    assert len(speed_range) == 2, "speed_range must be a list with 2 elements"
    if cf_e is None:
        # 将gap_range调整为从大到小
        if gap_range[-1] > gap_range[0]:
            gap_range = gap_range[::-1]
    else:
        gap_range = cf_e(speed=speed_range[0], **cf_param), cf_e(speed=speed_range[-1], **cf_param)
    gap_range = np.linspace(gap_range[0], gap_range[1], num=num_y)
    speed_range = np.linspace(speed_range[0], speed_range[1], num=num_x)

    dataX, dataY = np.meshgrid(speed_range, gap_range)

    resultAcc = [[cf_func(speed=dataX[j][i], gap=dataY[j][i], leaderV=lv, interval=dt, **cf_param)
                  for i in range(num_x)] for j in range(num_y)]  # 存储顺序与书写顺序相同, 储存v的结果

    resultDeltaV = [[resultAcc[j][i] * dt for i in range(num_x)] for j in range(num_y)]

    resultDeltaGap = [[lv * dt - (dataX[j][i] * dt) for i in range(num_x)] for j in range(num_y)]

    resultDeltaV = np.array(resultDeltaV)
    resultDeltaGap = np.array(resultDeltaGap)
    resultAcc = np.array(resultAcc)

    ax.quiver(dataX, dataY, resultDeltaV, resultDeltaGap, np.abs(resultAcc), pivot="tail", scale=1,
              units='inches', angles='xy', alpha=1, width=0.01, cmap=None, scale_units='xy')

    return resultDeltaV, resultDeltaGap, resultAcc


if __name__ == '__main__':
    q_IDM = QuiverInteract(
        cf_func=cf_IDM_acc,
        cf_e_func=cf_IDM_equilibrium,
        default={"s0": 2, "s1": 0, "v0": 33.3, "T": 1.6, "omega": 0.73, "d": 1.67, "delta": 4},
        range={"s0": [0, 5], "s1": [0, 5], "v0": [0, 40], "T": [0, 5], "omega": [0, 5], "d": [0, 5], "delta": [0, 10]},
        step={"s0": 0.1, "s1": 0.1, "v0": 0.1, "T": 0.1, "omega": 0.1, "d": 0.1, "delta": 1}
    )

    # q_Zhang = QuiverInteract(
    #     cf_func=cf_Zhang_acc,
    #     cf_e_func=cf_Zhang_equilibrium,
    #     default={"alpha": 0.5, "beta": 0.5, "v0": 30, "s0": 2, "T": 1.6},
    #     range={"alpha": [0, 10], "beta": [0, 10], "v0": [0, 40], "s0": [0, 5], "T": [0, 5]},
    #     step={"alpha": 0.1, "beta": 0.1, "v0": 0.1, "s0": 0.1, "T": 0.1}
    # )
