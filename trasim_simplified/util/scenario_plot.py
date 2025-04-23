# -*- coding: utf-8 -*-
# @time : 2025/4/14 17:34
# @Author : yzbyx
# @File : scenario_plot.py
# Software: PyCharm
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scienceplots  # type: ignore
import seaborn as sns
from matplotlib import pyplot as plt

from traj_process.processor.map_phrase.map_draw import MapDrawer
from trasim_simplified.core.constant import TrackInfo as TI, ScenarioMode

sns.reset_orig()
plt.style.use(['science', 'no-latex', 'cjk-sc-font'])

mm = 1 / 25.4  # mm转inch
fontsize = 10  # 7磅/pt/point

from hmmlearn.hmm import GMMHMM

# 读取纵向模型
model_path = fr"E:\BaiduSyncdisk\weaving-analysis\data\lon_gmm_hmm_model.pkl"
with open(model_path, 'rb') as f:
    lon_model: GMMHMM = pickle.load(f)


if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.road import Road


def bbox_traj_helper(x_c, y_c, yaw, ax, length=5, width=2,
                     color=None, linewidth=1., label=None, alpha=1):
    """车辆姿态的绘制"""
    # 车辆的四个角点
    corner_pos = np.array([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2]
    ])
    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    # 旋转并平移到车辆中心
    rotated_corners = corner_pos @ rotation_matrix.T + np.array([x_c, y_c])
    # 绘制车辆的矩形框，要闭合
    rotated_corners = np.vstack((rotated_corners, rotated_corners[0]))
    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color, linewidth=linewidth,
            label=label, alpha=alpha)
    ax.fill(rotated_corners[:, 0], rotated_corners[:, 1], color=color, alpha=alpha)


def get_fig_ax(scale=1, close_all=True):
    _width = 70 * mm * scale  # 图片宽度英寸
    _ratio = 5 / 7  # 图片长宽比
    figsize = (_width, _width * _ratio)

    if close_all:
        plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    return fig, ax


def plot_scenario(traj_s, traj_names, road: 'Road', fig_name):
    min_x_list = []
    min_y_list = []
    max_x_list = []
    max_y_list = []

    if isinstance(traj_s, pd.DataFrame):
        tem = []
        for track_id in traj_s[TI.trackId].unique():
            traj__ = traj_s[traj_s[TI.trackId] == track_id].sort_values(TI.frame)
            tem.append(traj__)
        traj_s = tem

    for traj__ in traj_s:
        if traj__ is None:
            continue
        min_x_list.append(np.min(traj__[TI.xCenterGlobal].values))
        max_x_list.append(np.max(traj__[TI.xCenterGlobal].values))
        min_y_list.append(np.min(traj__[TI.yCenterGlobal].values))
        max_y_list.append(np.max(traj__[TI.yCenterGlobal].values))
    min_x = min(min_x_list)
    max_x = max(max_x_list)
    min_y = min(min_y_list)
    max_y = max(max_y_list)

    # 设置坐标轴范围
    _width = 70 * mm * 2  # 图片宽度英寸
    _ratio = 5 / 14  # 图片长宽比
    figsize = (_width, _width * _ratio)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    road.draw(ax=ax)

    ax.set_xlim(min_x - 10, max_x + 10)
    ax.set_ylim(min_y - 5, max_y + 5)

    colors = sns.color_palette('Set1', n_colors=len(traj_s))

    # 绘制纵向行为模式散点
    for i, (traj_, name) in enumerate(zip(traj_s, traj_names)):
        if traj_ is None or len(traj_) == 0:
            continue
        # 计算纵向行为模式
        res = lon_model.predict(traj_[TI.localLonAcc].fillna(0).values.reshape(-1, 1))
        # 绘制纵向行为模式
        for j in range(len(res)):
            if res[j] == 0:
                ax.scatter(traj_[TI.xCenterGlobal].values[j], traj_[TI.yCenterGlobal].values[j],
                           color="blue", s=0.5)
            elif res[j] == 1:
                ax.scatter(traj_[TI.xCenterGlobal].values[j], traj_[TI.yCenterGlobal].values[j],
                           color="red", s=0.5)
            elif res[j] == 2:
                ax.scatter(traj_[TI.xCenterGlobal].values[j], traj_[TI.yCenterGlobal].values[j],
                           color="green", s=0.5)

        ax.plot(traj_[TI.xCenterGlobal].values, traj_[TI.yCenterGlobal].values, color="k", alpha=0.5,
                linewidth=0.2)

        frame_max = traj_[TI.frame].max()
        frame_min = traj_[TI.frame].min()
        for j, t in enumerate(np.linspace(frame_min, frame_max, 3).astype(int)):
            temp = traj_[traj_[TI.frame] == t]
            x_c = temp[TI.xCenterGlobal].values[0]
            y_c = temp[TI.yCenterGlobal].values[0]
            yaw = temp[TI.yaw].values[0]
            bbox_traj_helper(
                x_c=x_c, y_c=y_c, yaw=yaw,
                ax=ax, length=temp[TI.length].values[0], width=temp[TI.width].values[0],
                color=colors[i],
                linewidth=0.5, label=name if j == 0 else None
            )

    # 使用所有legend参数
    ax.legend(
        fontsize=fontsize, frameon=False,
        ncol=4, loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.savefig(
        fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{fig_name}_2d_traj.tif",
        dpi=300, pil_kwargs={"compression": "tiff_lzw"})

    fig, ax = get_fig_ax(scale=1)
    # 绘制各车辆的速度曲线
    for i, (traj_, name) in enumerate(zip(traj_s, traj_names)):
        if traj_ is None or len(traj_) == 0:
            continue
        ax.plot(traj_[TI.time].values, traj_[TI.speed].values, color=colors[i], label=name)
    # 绘制换道过线点垂线
    # for i, t in enumerate([frame]):
    #     ax.axvline(x=t * dt, color=colors[i], linestyle='--', label=f"换道点{i + 1}")
    ax.legend()
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("速度 (m/s)")

    fig.savefig(
        fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{fig_name}_speed.tif",
        dpi=300, pil_kwargs={"compression": "tiff_lzw"})

    # fig, ax = get_fig_ax(scale=1, close_all=False)
    # ax.plot(EV_traj["frame"].values * dt, EV_traj["heading"].values, color=colors[0], label="EV")
    #
    # fig, ax = get_fig_ax(scale=1, close_all=False)
    # ax.plot(EV_traj["frame"].values * dt, EV_traj["speed"].values, color=colors[1], label="EV")


def plot_scenario_twin(traj_s, traj_names, road: 'Road', fig_name, mode=ScenarioMode.NO_INTERACTION):
    min_x_list = []
    min_y_list = []
    max_x_list = []
    max_y_list = []

    if isinstance(traj_s, pd.DataFrame):
        tem = []
        for track_id in traj_s[TI.trackId].unique():
            traj__ = traj_s[traj_s[TI.trackId] == track_id].sort_values(TI.frame)
            tem.append(traj__)
        traj_s = tem

    for traj__ in traj_s:
        if traj__ is None:
            continue
        min_x_list.append(np.nanmin(traj__[TI.xCenterGlobal].values))
        max_x_list.append(np.nanmax(traj__[TI.xCenterGlobal].values))
        min_y_list.append(np.nanmin(traj__[TI.yCenterGlobal].values))
        max_y_list.append(np.nanmax(traj__[TI.yCenterGlobal].values))
    min_x = min(min_x_list)
    max_x = max(max_x_list)
    min_y = min(min_y_list)
    max_y = max(max_y_list)

    # 设置坐标轴范围
    _width = 70 * mm * 2  # 图片宽度英寸
    _ratio = 5 / 14  # 图片长宽比
    figsize = (_width, _width * _ratio)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    road.draw(ax=ax)

    # print(min_x, max_x, min_y, max_y)
    ax.set_xlim(min_x - 10, max_x + 10)
    ax.set_ylim(min_y - 5, max_y + 5)

    colors = sns.color_palette('tab20', n_colors=len(traj_s))

    # 绘制纵向行为模式散点
    for i, (traj_, name) in enumerate(zip(traj_s, traj_names)):
        if traj_ is None or len(traj_) == 0:
            continue
        alpha = 1 if name == "EV" else 0.2
        # 计算纵向行为模式
        res = lon_model.predict(traj_[TI.localLonAcc].fillna(0).values.reshape(-1, 1))
        # 绘制纵向行为模式
        for j in range(len(res)):
            if res[j] == 0:
                ax.scatter(traj_[TI.xCenterGlobal].values[j], traj_[TI.yCenterGlobal].values[j],
                           color="blue", s=0.5, alpha=alpha)
            elif res[j] == 1:
                ax.scatter(traj_[TI.xCenterGlobal].values[j], traj_[TI.yCenterGlobal].values[j],
                           color="red", s=0.5, alpha=alpha)
            elif res[j] == 2:
                ax.scatter(traj_[TI.xCenterGlobal].values[j], traj_[TI.yCenterGlobal].values[j],
                           color="green", s=0.5, alpha=alpha)

        ax.plot(traj_[TI.xCenterGlobal].values, traj_[TI.yCenterGlobal].values,
                color="k", alpha=alpha, linewidth=0.5)

        # 绘制ori轨迹
        if name == "EV":
            ax.plot(traj_[TI.xCenterGlobal + "_ori"].values,
                    traj_[TI.yCenterGlobal + "_ori"].values,
                    color="r", alpha=1, linewidth=0.5)

        frame_max = traj_[TI.frame].max()
        frame_min = traj_[TI.frame].min()
        for j, t in enumerate(np.linspace(frame_min, frame_max, 3).astype(int)):
            temp = traj_[traj_[TI.frame] == t]
            x_c = temp[TI.xCenterGlobal].values[0]
            y_c = temp[TI.yCenterGlobal].values[0]
            yaw = temp[TI.yaw].values[0]
            if j == 0:
                car_id = temp[TI.trackId].values[0]
                ax.text(x_c, y_c + 1, str(int(car_id)), ha="center", va="bottom",
                        fontsize=fontsize - 3)
            bbox_traj_helper(
                x_c=x_c, y_c=y_c, yaw=yaw,
                ax=ax, length=temp[TI.length].values[0], width=temp[TI.width].values[0],
                color=colors[i],
                linewidth=0.5, label=name if j == 0 else None,
                alpha=alpha
            )

    # 使用所有legend参数
    ax.legend(
        fontsize=fontsize, frameon=False,
        ncol=7, loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        # 行列间距缩小
        labelspacing=0.1,    # 标签间垂直间距
        handletextpad=0.1,   # 符号与标签水平间距
        # borderpad=1.0,       # 边框内边距
        columnspacing=0.1,   # 多列图例的列间距（若有多列）
    )

    ax.text(
        0.95, 0.95, fig_name.replace("_", "-"),
        fontsize=fontsize, color="black", ha='right', va='top',
        transform=ax.transAxes
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.savefig(
        fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{fig_name}_2d_traj.tif",
        dpi=300, pil_kwargs={"compression": "tiff_lzw"})

    plot_xy_info(traj_s, traj_names, colors, fig_name, TI.speed)
    # plot_xy_info(traj_s, traj_names, colors, fig_name, TI.acc)

    fig, ax = get_fig_ax(scale=1)
    # 绘制TTC曲线
    for i, (traj_, name) in enumerate(zip(traj_s, traj_names)):
        traj_: pd.DataFrame = traj_
        if traj_ is None or len(traj_) == 0 or name not in ["EV", "TR", "TP", "CR"]:
            continue
        z_order = 10 if name == "EV" else 1
        ax.plot(traj_[TI.time].values, traj_[TI.ttc].values, color=colors[i],
                label=name, zorder=z_order)

    for i, (traj_, name) in enumerate(zip(traj_s, traj_names)):
        traj_: pd.DataFrame = traj_
        if traj_ is None or len(traj_) == 0 or name not in ["EV", "TR", "TP", "CR"]:
            continue
        z_order = 10 if name == "EV" else 1
        ax.plot(traj_[TI.time].values, traj_[TI.ttc + "_ori"].values, color=colors[i],
                label=name + "$_{ori}$", zorder=z_order, linestyle="--", alpha=0.5)
    ax.legend(
        fontsize=fontsize, frameon=False,
        ncol=4, loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        # 行列间距缩小
        labelspacing=0.1,    # 标签间垂直间距
        handletextpad=0.1,   # 符号与标签水平间距
        # borderpad=1.0,       # 边框内边距
        columnspacing=0.1,   # 多列图例的列间距（若有多列）
    )
    # 绘制水平线TTC=1.3
    ax.axhline(y=1.3, color='r', linestyle='--', label="TTC=1.3s")
    # log纵坐标
    ax.set_yscale("log")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("TTC (s)")
    fig.savefig(
        fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{fig_name}_TTC.tif",
        dpi=300, pil_kwargs={"compression": "tiff_lzw"}
    )

    # 绘制EV横摆角
    fig, ax = get_fig_ax(scale=1)
    for i, (traj_, name) in enumerate(zip(traj_s, traj_names)):
        if name == "EV":
            ax.plot(traj_[TI.time].values, traj_[TI.yaw].values,
                    color=colors[i], label=name)
            ax.plot(traj_[TI.time].values, traj_[TI.yaw + "_ori"].values,
                    color=colors[i], label=name + "$_{ori}$",
                    linestyle="--", alpha=0.5)
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("横摆角 (rad)")
    fig.savefig(
        fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{fig_name}_yaw.tif",
        dpi=300, pil_kwargs={"compression": "tiff_lzw"}
    )


def plot_xy_info(traj_s, traj_names, colors, fig_name, info_name):
    fig, ax = get_fig_ax(scale=1)
    # 绘制各车辆的速度曲线
    for i, (traj_, name) in enumerate(zip(traj_s, traj_names)):
        traj_: pd.DataFrame = traj_
        if traj_ is None or len(traj_) == 0 or name not in ["EV", "TR", "TP", "CP", "CR"]:
            continue
        z_order = 10 if name == "EV" else 1
        ax.plot(traj_[TI.time].values,
                traj_[info_name].values * np.cos(traj_[TI.yaw]),
                color=colors[i],
                label=name, zorder=z_order)
        if name == "EV":
            # 绘制ev的真实速度曲线
            ax.plot(traj_[TI.time].values,
                    traj_[info_name + "_ori"].values * np.cos(traj_[TI.yaw + "_ori"].values),
                    color=colors[i], label="EV$_{ori}$",
                    linestyle="--", linewidth=1, zorder=0, alpha=0.5)
    ax.legend(
        fontsize=fontsize, frameon=False,
        ncol=4, loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        # 行列间距缩小
        labelspacing=0.1,    # 标签间垂直间距
        handletextpad=0.1,   # 符号与标签水平间距
        # borderpad=1.0,       # 边框内边距
        columnspacing=0.1,   # 多列图例的列间距（若有多列）
    )
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("纵向速度 (m/s)")
    fig.savefig(
        fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{fig_name}_{info_name}_x.tif",
        dpi=300, pil_kwargs={"compression": "tiff_lzw"})

    # 绘制各车辆的速度曲线
    fig, ax = get_fig_ax(scale=1)
    for i, (traj_, name) in enumerate(zip(traj_s, traj_names)):
        traj_: pd.DataFrame = traj_
        if traj_ is None or len(traj_) == 0 or name not in ["EV", "TR", "TP", "CP", "CR"]:
            continue
        z_order = 10 if name == "EV" else 1
        ax.plot(traj_[TI.time].values, traj_[info_name].values * np.sin(traj_[TI.yaw]),
                color=colors[i],
                label=name, zorder=z_order)
        if name == "EV":
            # 绘制ev的真实速度曲线
            ax.plot(traj_[TI.time].values,
                    traj_[info_name + "_ori"].values * np.sin(traj_[TI.yaw + "_ori"].values),
                    color=colors[i], label="EV$_{ori}$",
                    linestyle="--", linewidth=1, zorder=0, alpha=0.5)
    ax.legend(
        fontsize=fontsize, frameon=False,
        ncol=4, loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        # 行列间距缩小
        labelspacing=0.1,    # 标签间垂直间距
        handletextpad=0.1,   # 符号与标签水平间距
        # borderpad=1.0,       # 边框内边距
        columnspacing=0.1,   # 多列图例的列间距（若有多列）
    )
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("横向速度 (m/s)")
    fig.savefig(
        fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{fig_name}_{info_name}_y.tif",
        dpi=300, pil_kwargs={"compression": "tiff_lzw"}
    )
