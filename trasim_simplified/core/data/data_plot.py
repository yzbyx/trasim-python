# -*- coding = uft-8 -*-
# @Time : 2023-03-31 15:50
# @Author : yzbyx
# @File : data_plot.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Union, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mc
import matplotlib.collections as mcoll
from matplotlib.colorbar import Colorbar

from trasim_simplified.core.data.data_processor import Info as P_Info
from trasim_simplified.core.data.data_container import Info as C_Info

if TYPE_CHECKING:
    from trasim_simplified.core.frame.frame_abstract import FrameAbstract


class Plot:
    def __init__(self, frame_abstract: 'FrameAbstract'):
        self.frame = frame_abstract
        self.container = self.frame.data_container
        self.processor = self.frame.data_processor

    def basic_plot(self, index=0, axes: plt.Axes=None, fig: plt.Figure=None):
        """绘制车辆index"""
        time_ = np.arange(self.frame.warm_up_step, self.frame.sim_step) * self.frame.dt

        if axes is None or fig is None:
            fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.5), layout="constrained")

        ax = axes[0, 0]
        self.custom_plot(ax, "time(s)", "speed(m/s)", time_, self.container.speed_data[:, index], data_label=f"index={index}")

        ax = axes[0, 1]
        self.custom_plot(ax, "speed(m/s)", "gap(m)",
                         self.container.speed_data[:, index], self.container.gap_data[:, index], data_label=f"index={index}")

        ax = axes[1, 0]
        self.custom_plot(ax, "dv(m/s)", "gap(m)",
                         self.container.dv_data[:, index], self.container.gap_data[:, index], data_label=f"index={index}")

        ax = axes[1, 1]
        self.custom_plot(ax, "time(s)", "acc(m/s^2)", time_, self.container.acc_data[:, index], data_label=f"index={index}")

        if P_Info.safe_tit in self.processor.info:
            ax = axes[0, 2]
            self.custom_plot(ax, "time(s)", "tit(s)", time_,
                             self.processor.safe_result[P_Info.safe_tit][:, index], data_label=f"index={index}")

        if P_Info.safe_picud in self.processor.info:
            ax = axes[1, 2]
            self.custom_plot(ax, "time(s)", "picud(m)", time_,
                             self.processor.safe_result[P_Info.safe_picud][:, index], data_label=f"index={index}")

        ax = axes[2][0]
        self.custom_plot(ax, "time(s)", "gap(m)", time_, self.container.gap_data[:, index], data_label=f"index={index}")

    def spatial_time_plot(self, index=0, color_data=None, color_bar_name=None):
        if color_data is None:
            color_data = self.container.speed_data
            color_bar_name = C_Info.v
        assert color_data is not None and color_bar_name is not None, "color_data或color_bar_name不能为None"

        fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout="constrained")
        ax: plt.Axes = ax
        fig: plt.Figure = fig
        ax.set_xlabel("time(s)")
        ax.set_ylabel("location")
        value_range = (np.min(color_data), np.max(color_data))
        cmap = plt.get_cmap('rainbow')
        for time__, temp__, index_, pos in self.frame.data_processor.data_shear(self.container.pos_data):
            if pos[1] - pos[0] > 1:
                self._lines_color(ax, time__, temp__, color_data[:, index_][pos[0]: pos[1]], value_range,
                                  cmap, line_width=0.2 if index != index_ else 1)

        cb: Colorbar = fig.colorbar(ScalarMappable(mc.Normalize(vmin=value_range[0], vmax=value_range[1]), cmap))
        cb.set_label(color_bar_name)

    @staticmethod
    def show():
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

    @staticmethod
    def custom_plot(ax: plt.Axes, x_label: str, y_label: str,
                    x_data: Union[list[Sequence], Sequence], y_data: Union[list[Sequence], Sequence], *args,
                    data_label: Union[list[str], Union[str, None]]=None, **kwargs):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if not isinstance(x_data, list): x_data = [x_data]
        if not isinstance(y_data, list): y_data = [y_data]
        if not isinstance(data_label, list): data_label = [data_label]
        for i, x_data_ in enumerate(x_data):
            ax.plot(x_data_, y_data[i], *args, label=data_label[i], **kwargs)
        ax.legend()
