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
    from trasim_simplified.core.frame.lane_abstract import LaneAbstract


class Plot:
    def __init__(self, frame_abstract: 'LaneAbstract'):
        self.frame = frame_abstract
        self.container = self.frame.data_container
        self.processor = self.frame.data_processor

    def basic_plot(self, index=0, axes: plt.Axes = None, fig: plt.Figure = None):
        """绘制车辆index"""
        time_ = self.container.get_data(index, C_Info.time)

        if axes is None or fig is None:
            fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.5), layout="constrained")

        ax = axes[0, 0]
        self.custom_plot(ax, "time(s)", "speed(m/s)", time_, self.container.get_data(index, C_Info.v),
                         data_label=f"index={index}")

        ax = axes[0, 1]
        self.custom_plot(ax, "speed(m/s)", "gap(m)",
                         self.container.get_data(index, C_Info.v), self.container.get_data(index, C_Info.gap),
                         data_label=f"index={index}")

        ax = axes[1, 0]
        self.custom_plot(ax, "dv(m/s)", "gap(m)",
                         self.container.get_data(index, C_Info.dv), self.container.get_data(index, C_Info.gap),
                         data_label=f"index={index}")

        ax = axes[1, 1]
        self.custom_plot(ax, "time(s)", "acc(m/s^2)", time_, self.container.get_data(index, C_Info.a),
                         data_label=f"index={index}")

        if P_Info.safe_tit in self.container.save_info:
            ax = axes[0, 2]
            self.custom_plot(ax, "time(s)", "tit(s)", time_,
                             self.container.get_data(index, C_Info.safe_tit), data_label=f"index={index}")

        if P_Info.safe_picud in self.container.save_info:
            ax = axes[1, 2]
            self.custom_plot(ax, "time(s)", "picud(m)", time_,
                             self.container.get_data(index, C_Info.safe_picud), data_label=f"index={index}")

        ax = axes[2][0]
        self.custom_plot(ax, "time(s)", "gap(m)", time_,
                         self.container.get_data(index, C_Info.gap), data_label=f"index={index}")

    def spatial_time_plot(self, index=0, color_info_name=None):
        if color_info_name is None:
            color_data = self.container.data_df[C_Info.v]
            color_bar_name = C_Info.v
        else:
            color_data = self.container.data_df[color_info_name]
            color_bar_name = color_info_name

        fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout="constrained")
        ax: plt.Axes = ax
        fig: plt.Figure = fig
        ax.set_xlabel("time(s)")
        ax.set_ylabel("location")
        value_range = (np.min(color_data), np.max(color_data))
        cmap = plt.get_cmap('rainbow')
        for id_ in [car.ID for car in self.container.get_total_car_has_data()]:
            color_data_single = self.container.get_data(id_, color_info_name)
            for time__, temp__, pos in self.frame.data_processor.data_shear(C_Info.x, id_):
                if pos[1] - pos[0] > 1:
                    self._lines_color(ax, time__, temp__, color_data_single[pos[0]: pos[1]],
                                      value_range, cmap, line_width=0.2 if index != id_ else 1)

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

        if value_range[1] - value_range[0] != 0:
            color_value = (color_value - value_range[0]) / (value_range[1] - value_range[0])
        colors = cmap(color_value)
        lc = mcoll.LineCollection(seg, colors=colors, linewidths=line_width)

        ax.add_collection(lc)
        ax.autoscale(True)
        return lc

    @staticmethod
    def custom_plot(ax: plt.Axes, x_label: str, y_label: str,
                    x_data: Union[list[Sequence], Sequence], y_data: Union[list[Sequence], Sequence], *args,
                    data_label: Union[list[str], Union[str, None]] = None, **kwargs):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if not isinstance(x_data, list): x_data = [x_data]
        if not isinstance(y_data, list): y_data = [y_data]
        if not isinstance(data_label, list): data_label = [data_label]
        for i, x_data_ in enumerate(x_data):
            ax.plot(x_data_, y_data[i], *args, label=data_label[i], **kwargs)
        ax.legend()
