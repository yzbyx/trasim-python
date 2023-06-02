# -*- coding = uft-8 -*-
# @Time : 2023-03-31 15:50
# @Author : yzbyx
# @File : data_plot.py
# @Software : PyCharm
from typing import Union, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mc
import matplotlib.collections as mcoll
from matplotlib.colorbar import Colorbar

from trasim_simplified.core.data.data_processor import Info as P_Info, DataProcessor
from trasim_simplified.core.data.data_container import Info as C_Info


class Plot:
    @staticmethod
    def basic_plot(id_=0, lane_id=-1, axes: plt.Axes = None, fig: plt.Figure = None, data_df: pd.DataFrame = None):
        """绘制车辆index"""
        if lane_id >= 0:
            data_df = data_df[data_df[C_Info.lane_id] == lane_id]
        car_df = data_df[data_df[C_Info.id] == id_]
        time_ = car_df[C_Info.time]

        if axes is None or fig is None:
            fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.5), layout="constrained")

        ax = axes[0, 0]
        Plot.custom_plot(ax, "time(s)", "speed(m/s)", time_, car_df[C_Info.v],
                         data_label=f"index={id_}")

        ax = axes[0, 1]
        Plot.custom_plot(ax, "speed(m/s)", "gap(m)",
                         car_df[C_Info.v], car_df[C_Info.gap],
                         data_label=f"index={id_}")

        ax = axes[1, 0]
        Plot.custom_plot(ax, "dv(m/s)", "gap(m)",
                         car_df[C_Info.dv], car_df[C_Info.gap],
                         data_label=f"index={id_}")

        ax = axes[1, 1]
        Plot.custom_plot(ax, "time(s)", "acc(m/s^2)", time_, car_df[C_Info.a],
                         data_label=f"index={id_}")

        if P_Info.safe_tit in car_df.columns:
            ax = axes[0, 2]
            Plot.custom_plot(ax, "time(s)", "tit(s)", time_,
                             car_df[C_Info.safe_tit], data_label=f"index={id_}")

        if P_Info.safe_picud in car_df.columns:
            ax = axes[1, 2]
            Plot.custom_plot(ax, "time(s)", "picud(m)", time_,
                             car_df[C_Info.safe_picud], data_label=f"index={id_}")

        ax = axes[2][0]
        Plot.custom_plot(ax, "time(s)", "gap(m)", time_,
                         car_df[C_Info.gap], data_label=f"index={id_}")

    @staticmethod
    def spatial_time_plot(car_id=0, lane_id=0, color_info_name=None, data_df: pd.DataFrame = None, single_plot=False):
        data_df = data_df[data_df[C_Info.lane_id] == lane_id]
        data_df = data_df.sort_values(by=[C_Info.id, C_Info.time]).reset_index(drop=True)
        if color_info_name is None:
            color_data = data_df[C_Info.v]
            color_bar_name = C_Info.v
        else:
            color_data = data_df[color_info_name]
            color_bar_name = color_info_name

        fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout="constrained")
        ax: plt.Axes = ax
        fig: plt.Figure = fig
        ax.set_title(f"lane_id: {lane_id}")
        ax.set_xlabel("time(s)")
        ax.set_ylabel("location")
        value_range = (np.min(color_data), np.max(color_data))
        cmap = plt.get_cmap('rainbow')
        for id_ in data_df[C_Info.id].unique():
            if car_id != id_ and single_plot: continue
            car_info = data_df[data_df[C_Info.id] == id_]
            car_info = car_info.sort_values(by=[C_Info.time]).reset_index(drop=True)
            color_data_single = car_info[color_info_name]
            temp_ = car_info[C_Info.x]
            pos_ = car_info[C_Info.x]
            time_ = car_info[C_Info.time]
            step_ = car_info[C_Info.step]
            for time__, temp__, pos in DataProcessor.data_shear(temp_, pos_, time_, step_):
                if pos[1] - pos[0] > 1:
                    Plot._lines_color(ax, time__, temp__, color_data_single[pos[0]: pos[1]],
                                      value_range, cmap, line_width=0.2 if car_id != id_ else 1)

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
