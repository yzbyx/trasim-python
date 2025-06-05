# -*- coding = uft-8 -*-
# @time : 2023-03-31 15:50
# @Author : yzbyx
# @File : data_plot.py
# @Software : PyCharm
from typing import Union, Sequence, Callable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mc
import matplotlib.collections as mcoll
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
from matplotlib.colors import Colormap


from trasim_simplified.core.data.data_processor import Info as P_Info, DataProcessor, DetectLoopInfo as DLI
try:
    from trasim_simplified.core.constant import TrackInfo as C_Info
except ImportError:
    print("Tools not found, use trasim_simplified.core.data.data_container.Info")
    from trasim_simplified.core.data.data_container import Info as C_Info


def get_single_ax_fig() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(10.5, 7.5), layout="constrained")


class Plot:
    @staticmethod
    def basic_plot(
        data_df: pd.DataFrame,
        id_: Union[int, list[int]] = 0,
        lane_id=-1,
        axes: np.ndarray[plt.Axes] = None,
        time_range=None,
    ):
        """绘制车辆index"""
        if isinstance(id_, int):
            id_list = [id_]
        else:
            id_list = id_

        if lane_id >= 0:
            data_df = data_df[data_df[C_Info.lane_add_num] == lane_id]

        if time_range is not None:
            data_df = data_df[
                (data_df[C_Info.time] >= time_range[0])
                & (data_df[C_Info.time] < time_range[1])
            ]

        car_dfs = [data_df[data_df[C_Info.id] == id_] for id_ in id_list]

        if axes is None:
            fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.5), layout="constrained")

        ax = axes[0, 0]
        Plot.custom_plot(
            ax,
            "time(s)",
            "speed(m/s)",
            [car_df[C_Info.time] for car_df in car_dfs],
            [car_df[C_Info.v] for car_df in car_dfs],
            data_label=[f"index={id_}" for id_ in id_list],
        )

        ax = axes[0, 1]
        Plot.custom_plot(
            ax,
            "speed(m/s)",
            "gap(m)",
            [car_df[C_Info.v] for car_df in car_dfs],
            [car_df[C_Info.gap] for car_df in car_dfs],
            data_label=[f"index={id_}" for id_ in id_list],
        )

        ax = axes[1, 0]
        Plot.custom_plot(
            ax,
            "dv(m/s)",
            "gap(m)",
            [car_df[C_Info.dv] for car_df in car_dfs],
            [car_df[C_Info.gap] for car_df in car_dfs],
            data_label=[f"index={id_}" for id_ in id_list],
        )

        ax = axes[1, 1]
        Plot.custom_plot(
            ax,
            "time(s)",
            "acc(m/s^2)",
            [car_df[C_Info.time] for car_df in car_dfs],
            [car_df[C_Info.a] for car_df in car_dfs],
            data_label=[f"index={id_}" for id_ in id_list],
        )

        if P_Info.safe_tit in car_dfs[0].columns:
            ax = axes[0, 2]
            Plot.custom_plot(
                ax,
                "time(s)",
                "tit(s)",
                [car_df[C_Info.time] for car_df in car_dfs],
                [car_df[C_Info.safe_tit] for car_df in car_dfs],
                data_label=[f"index={id_}" for id_ in id_list],
            )

        if P_Info.safe_picud in car_dfs[0].columns:
            ax = axes[1, 2]
            Plot.custom_plot(
                ax,
                "time(s)",
                "picud(m)",
                [car_df[C_Info.time] for car_df in car_dfs],
                [car_df[C_Info.safe_picud_KK] for car_df in car_dfs],
                data_label=[f"index={id_}" for id_ in id_list],
            )

        ax = axes[2][0]
        Plot.custom_plot(
            ax,
            "time(s)",
            "gap(m)",
            [car_df[C_Info.time] for car_df in car_dfs],
            [car_df[C_Info.gap] for car_df in car_dfs],
            data_label=[f"index={id_}" for id_ in id_list],
        )

        return axes

    @staticmethod
    def add_plot_2D(
        ax: Optional[plt.Axes],
        func_: Callable,
        x_step: float,
        x_range: Optional[list[float, float]] = None,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout="constrained")
        if x_range is None:
            lines: list[Line2D] = ax.get_lines()
            x_data = []
            for i in range(len(lines)):
                x_data.extend(list(lines[i].get_data()[0]))
            x_range = (min(x_data), max(x_data))
        range_ = np.arange(x_range[0], x_range[1], x_step)
        y_ = func_(range_)
        ax.plot(range_, y_, **kwargs)
        return ax

    @staticmethod
    def spatial_time_plot(
        data_df: pd.DataFrame,
        car_id=-1,
        lane_add_num=0,
        color_info_name=None,
        single_plot=False,
        color_lambda_: Optional[Callable] = None,
        color_value_remove_outliers=False,
        base_line_width: float = 0.2,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        color_bar=True,
        cmap_name="rainbow",
        frame_rate=10,
    ):
        if C_Info.lane_add_num in data_df.columns:
            data_df = data_df[data_df[C_Info.lane_add_num] == lane_add_num]

        if C_Info.time not in data_df.columns:
            print(f"time not in data_df.columns, calculate it... (frame_rate={frame_rate})")
            data_df[C_Info.time] = data_df[C_Info.step] / frame_rate
        data_df = data_df.sort_values(by=[C_Info.id, C_Info.time]).reset_index(
            drop=True
        )
        if color_info_name is None:
            color_data = data_df[C_Info.v]
            color_info_name = C_Info.v
        else:
            color_data = data_df[color_info_name]
        color_bar_name = color_info_name

        if color_lambda_ is not None:
            color_data = color_data.apply(color_lambda_)
            data_df[color_info_name] = color_data

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout="constrained")
            ax: plt.Axes = ax
            fig: plt.Figure = fig
            ax.set_title(f"lane_add_num: {lane_add_num}")
            ax.set_xlabel("time(s)")
            ax.set_ylabel("location")

        c_data = (
            color_data.to_numpy()
            if not color_value_remove_outliers
            else Plot.remove_outliers(color_data.to_numpy())
        )
        value_range = (np.min(c_data), np.max(c_data))
        cmap: Colormap = plt.get_cmap(cmap_name)
        cmap.set_over("w")
        cmap.set_under("k")
        for id_ in data_df[C_Info.id].unique():
            if car_id != id_ and single_plot:
                continue
            car_info = data_df[data_df[C_Info.id] == id_]
            car_info = car_info.sort_values(by=[C_Info.time]).reset_index(drop=True)
            color_data_single = car_info[color_info_name]
            temp_ = car_info[C_Info.x]
            pos_ = car_info[C_Info.x]
            time_ = car_info[C_Info.time]
            step_ = car_info[C_Info.step]
            for time__, temp__, pos in DataProcessor.data_shear(
                temp_, pos_, time_, step_
            ):
                if pos[1] - pos[0] > 1:
                    Plot._lines_color(
                        ax,
                        time__,
                        temp__,
                        color_data_single[pos[0] : pos[1]],
                        value_range,
                        cmap,
                        line_width=base_line_width
                        if car_id != id_
                        else base_line_width * 5,
                    )

        if color_bar:
            cb: Colorbar = fig.colorbar(
                ScalarMappable(
                    mc.Normalize(vmin=value_range[0], vmax=value_range[1]), cmap
                ),
                ax=ax
            )
            cb.set_label(color_bar_name)
        return fig, ax

    @staticmethod
    def remove_outliers(data: np.ndarray):
        data = data[~np.isnan(data)]
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        iqr = q3 - q1
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        result = data[np.where((data >= fence_low) & (data <= fence_high))]
        return result

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def _lines_color(
        ax, data_x, data_y, color_value, value_range, cmap, line_width=0.2
    ):
        """绘制多彩折线"""
        data_x = np.array(data_x).reshape(-1)
        data_y = np.array(data_y).reshape(-1)
        points = [(x, y) for x, y in zip(data_x, data_y)]
        color_value = np.array(color_value).reshape(-1)
        color_value = color_value[:-1]
        seg = np.array([(a, b) for a, b in zip(points[:-1], points[1:])])

        if value_range[1] - value_range[0] != 0:
            color_value = (color_value - value_range[0]) / (
                value_range[1] - value_range[0]
            )
        colors = cmap(color_value)
        lc = mcoll.LineCollection(seg, colors=colors, linewidths=line_width)

        ax.add_collection(lc)
        ax.autoscale(True)
        return lc

    @staticmethod
    def custom_plot(
        ax: plt.Axes,
        x_label: str,
        y_label: str,
        x_data: Union[list[Sequence], Sequence],
        y_data: Union[list[Sequence], Sequence],
        *args,
        data_label: Union[list[str], Union[str, None]] = None,
        **kwargs,
    ):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if not isinstance(x_data, list):
            x_data = [x_data]
        if not isinstance(y_data, list):
            y_data = [y_data]
        if not isinstance(data_label, list):
            data_label = [data_label]
        for i, x_data_ in enumerate(x_data):
            ax.plot(x_data_, y_data[i], *args, label=data_label[i], **kwargs)
        ax.legend()
        return ax

    @staticmethod
    def plot_density_map(
        df: pd.DataFrame,
        lane_id: int,
        dt: float,
        d_step: int,
        d_space: float,
        step_range: Optional[list[int, int]] = None,
        space_range: Optional[list[float, float]] = None,
    ):
        """使用HCM方法"""
        fig, axes = plt.subplots(1, 3, figsize=(21, 5), layout="constrained")
        titles = [DLI.HCM_kA, DLI.HCM_qA, DLI.HCM_vA]
        for i, ax in enumerate(axes):
            ax.set_xlabel("time block")
            ax.set_ylabel("space block")
            ax: plt.Axes = ax
            ax.set_title(titles[i])

        full_df = df.copy()

        df = df[df[C_Info.lane_add_num] == lane_id]

        k = []
        q = []
        v = []

        if step_range is None:
            step_range = (df[C_Info.step].min(), df[C_Info.step].max())

        time_sections = np.arange(step_range[0], step_range[1] + 1, d_step)
        time_sections_zip = list(zip(time_sections[:-1], time_sections[1:]))

        if space_range is None:
            space_range = (df[C_Info.x].min(), df[C_Info.x].max())

        space_sections = np.arange(space_range[0], space_range[1] + 1e-6, d_space)
        space_sections_zip = list(zip(space_sections[:-1], space_sections[1:]))

        for start, _ in space_sections_zip:
            _, result = DataProcessor.aggregate_as_detect_loop(
                full_df,
                lane_id,
                lane_length=np.inf,
                pos=start,
                width=d_space,
                dt=dt,
                d_step=d_step,
                step_range=step_range,
            )
            k.append(result[DLI.HCM_kA])
            q.append(result[DLI.HCM_qA])
            v.append(result[DLI.HCM_vA])

        for i, ax in enumerate(axes):
            data = np.array(k if i == 0 else (q if i == 1 else v))
            im = ax.imshow(data)
            ax.set_xticks(list(range(data.shape[1])))
            ax.set_yticks(list(range(data.shape[0])))
            ax.set_xticklabels([f"{start:.0f}s:{end:.0f}s" for start, end in time_sections_zip])
            ax.set_yticklabels([f"{start:.0f}m:{end:.0f}m" for start, end in space_sections_zip])
            fig.colorbar(im)
        return fig, axes

    @staticmethod
    def two_dim_plot(df: pd.DataFrame, x_index: str, y_index: str,
                     fig: plt.Figure = None, ax: plt.Axes = None, **kwargs):
        if ax is None or fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5), layout="constrained")
        ax: plt.Axes = ax
        ax.set_xlabel(x_index)
        ax.set_ylabel(y_index)
        for id_ in df[C_Info.trackId].unique():
            target = df[df[C_Info.trackId] == id_]
            ax.plot(target[x_index], target[y_index], label=id_, **kwargs)
        fig.legend()
        return fig, ax


if __name__ == '__main__':
    fig, ax = plt.subplots(layout="constrained")
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    im = ax.imshow([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    fig.colorbar(im)
    plt.show()
