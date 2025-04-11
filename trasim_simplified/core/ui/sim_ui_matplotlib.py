# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:02
# @Author : yzbyx
# @File : sim_ui_matplotlib.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from typing import TYPE_CHECKING, Optional, Union, Tuple

from trasim_simplified.core.constant import MARKING_TYPE

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract
    from trasim_simplified.core.frame.micro.road import Road
    from trasim_simplified.core.agent import Vehicle, Game_A_Vehicle


class UI2DMatplotlib:
    def __init__(self, frame_abstract: Union['LaneAbstract', 'Road']):
        self.frame = frame_abstract
        self.frame_rate = -1
        self.width_base = 1000
        self.x_scale = 1.5  # 水平方向缩放
        self.y_scale = 2.0  # 垂直方向缩放
        self.single_height = 20
        self.base_line_factor = 2
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.road_patches = []  # 存储道路相关的patches
        self.text_list: Optional[list[plt.Text]] = []
        self.step_text = None

        if not hasattr(self.frame, "lane_list"):
            self.lane_list = [self.frame]
        else:
            self.lane_list = self.frame.lane_list

    @staticmethod
    def normalize_color(color: Tuple[int, int, int]) -> tuple[float, ...]:
        """将0-255范围的RGB颜色值转换为0-1范围"""
        return tuple(c / 255.0 for c in color)

    def focus_on(self, veh: 'Vehicle'):
        """
        设置焦点车辆
        :param veh: 车辆对象
        """
        if self.ax is not None:
            # 设置x轴和y轴的范围
            x_min = veh.x - 100
            x_max = veh.x + 100
            y_min = veh.y - 10
            y_max = veh.y + 10

            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)

    def plot_pred_traj(self):
        """
        绘制预测轨迹
        """
        for lane in self.lane_list:
            for car in lane.car_list:
                pred_traj = car.pred_traj
                if pred_traj is not None:
                    x = [point.x for point in pred_traj]
                    y = [point.y for point in pred_traj]
                    self.ax.plot(x, y, color='blue', linestyle='-', linewidth=1)

    def plot_hist_traj(self):
        """
        绘制历史轨迹
        """
        for lane in self.lane_list:
            for car in lane.car_list:
                hist_traj = car.hist_traj
                if hist_traj is not None:
                    x = [point.x for point in hist_traj]
                    y = [point.y for point in hist_traj]
                    self.ax.plot(x, y, color='red', linestyle='-', linewidth=1)

    def ui_init(self, caption="traffic simulation", frame_rate=-1):
        """
        初始化UI
        :param caption: 窗口标题
        :param frame_rate: 帧率
        """
        self.frame_rate = frame_rate

        self.fig, self.ax = plt.subplots(figsize=(20, 6))
        self.ax.set_title(caption)
        # 等比例
        self.ax.set_aspect('equal', adjustable='box')
        plt.ion()  # 开启交互模式

        self.draw_road()  # 只在初始化时绘制道路
        self.ui_update()  # 更新显示

    def draw_road(self):
        """绘制道路，只在初始化时调用一次"""
        self.ax.set_facecolor('green')

        for lane in self.lane_list:
            # 绘制车道背景
            rect = Rectangle((0, lane.y_right), lane.lane_length,
                             lane.width, facecolor='gray', alpha=1)
            self.ax.add_patch(rect)
            self.road_patches.append(rect)

            # 绘制车道线
            if lane.marking_type is None or len(lane.marking_type[1]) == 0:
                line1, = self.ax.plot([0, lane.lane_length], [lane.y_left, lane.y_left],
                                      color='yellow', linewidth=1)
                line2, = self.ax.plot([0, lane.lane_length], [lane.y_right, lane.y_right],
                                      color='yellow', linewidth=1)
                self.road_patches.extend([line1, line2])
            else:
                if lane.marking_type is None:
                    line1, = self.ax.plot([0, lane.lane_length], [lane.y_left, lane.y_left],
                                          color='yellow', linewidth=1)
                    line2, = self.ax.plot([0, lane.lane_length], [lane.y_right, lane.y_right],
                                          color='yellow', linewidth=1)
                    self.road_patches.extend([line1, line2])
                else:
                    for i in range(0, len(lane.marking_type[1])):
                        # 绘制左车道线
                        marker_type: MARKING_TYPE = lane.marking_type[1][i][0]
                        start = lane.marking_type[0][i]
                        end = lane.marking_type[0][i + 1]
                        color = 'white' if marker_type == MARKING_TYPE.DASHED else 'yellow'
                        linestyle = '--' if marker_type == MARKING_TYPE.DASHED else '-'
                        line1, = self.ax.plot([start, end], [lane.y_left, lane.y_left],
                                              color=color, linewidth=1, linestyle=linestyle)
                        self.road_patches.append(line1)

                        # 绘制右车道线
                        marker_type: MARKING_TYPE = lane.marking_type[1][i][1]
                        start = lane.marking_type[0][i]
                        end = lane.marking_type[0][i + 1]
                        color = 'white' if marker_type == MARKING_TYPE.DASHED else 'yellow'
                        linestyle = '--' if marker_type == MARKING_TYPE.DASHED else '-'
                        line2, = self.ax.plot([start, end], [lane.y_right, lane.y_right],
                                              color=color, linewidth=1, linestyle=linestyle)
                        self.road_patches.append(line2)

    def ui_update(self):
        """更新UI显示，只更新动态内容"""
        # 清除之前的车辆patches
        for patch in self.ax.patches:
            if patch not in self.road_patches:
                patch.remove()
        for line in self.ax.lines:
            if line not in self.road_patches:
                line.remove()
        for text in self.text_list:
            text.remove()
        self.text_list.clear()

        # 显示步数
        if self.step_text is None:
            self.step_text = self.ax.text(0, 0, f"steps: {self.frame.step_}",
                                          fontsize=12, color='black')
        else:
            self.step_text.set_text(f"steps: {self.frame.step_}")

        # 绘制车辆
        for lane in self.lane_list:
            for car in lane.car_list:
                polygon = car.plot_car(self.ax)
                # 将车辆颜色从0-255范围转换为0-1范围
                normalized_color = self.normalize_color(car.color)
                # car_patch = Polygon(polygon, facecolor=normalized_color, alpha=0.5,
                #                     edgecolor='black', linewidth=1, closed=True)
                # self.ax.add_patch(car_patch)
                # 加上ID
                text = self.ax.text(car.x, car.y, str(car.ID), fontsize=12,
                                    color='white', ha='center', va='center')
                self.text_list.append(text)
                # 如果换道，且为AV，则绘制换道轨迹
                if car.lane_changing and hasattr(car, "lc_traj"):
                    if car.lc_traj is None:
                        continue
                    self.ax.plot(car.lc_traj[:, 0], car.lc_traj[:, 3],
                                 color='blue', linestyle='--', linewidth=1)

        plt.pause(0.01)  # 短暂暂停以更新显示
        if self.frame_rate > 0:
            plt.pause(1 / self.frame_rate)
