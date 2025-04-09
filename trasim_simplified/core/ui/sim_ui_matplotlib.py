# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:02
# @Author : yzbyx
# @File : sim_ui_matplotlib.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from typing import TYPE_CHECKING, Optional, Union, Tuple
import numpy as np

from trasim_simplified.core.constant import MARKING_TYPE

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract
    from trasim_simplified.core.frame.micro.road import Road


class UI2DMatplotlib:
    def __init__(self, frame_abstract: Union['LaneAbstract', 'Road']):
        self.frame = frame_abstract
        self.frame_rate = -1
        self.width_base = 1000
        self.x_scale = 1.5  # 水平方向缩放
        self.y_scale = 2.0  # 垂直方向缩放
        self.single_height = 20
        self.base_line_factor = 2
        self.fig = None
        self.ax = None
        self.road_patches = []  # 存储道路相关的patches
        self.text = None

        if not hasattr(self.frame, "lane_list"):
            self.lane_list = [self.frame]
        else:
            self.lane_list = self.frame.lane_list

    @staticmethod
    def normalize_color(color: Tuple[int, int, int]) -> tuple[float, ...]:
        """将0-255范围的RGB颜色值转换为0-1范围"""
        return tuple(c / 255.0 for c in color)

    def ui_init(self, caption="traffic simulation", frame_rate=-1):
        """
        初始化UI
        :param caption: 窗口标题
        :param frame_rate: 帧率
        """
        self.frame_rate = frame_rate

        self.fig, self.ax = plt.subplots(figsize=(20, 6))
        self.ax.set_title(caption)
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
                        line1, = self.ax.plot([start, end], [lane.y_left, lane.y_left],
                                              color=color, linewidth=1)
                        self.road_patches.append(line1)

                        # 绘制右车道线
                        marker_type: MARKING_TYPE = lane.marking_type[1][i][1]
                        start = lane.marking_type[0][i]
                        end = lane.marking_type[0][i + 1]
                        color = 'white' if marker_type == MARKING_TYPE.DASHED else 'yellow'
                        line2, = self.ax.plot([start, end], [lane.y_right, lane.y_right],
                                              color=color, linewidth=1)
                        self.road_patches.append(line2)

    def ui_update(self):
        """更新UI显示，只更新动态内容"""
        # 清除之前的车辆patches
        for patch in self.ax.patches:
            if patch not in self.road_patches:
                patch.remove()

        # 显示步数
        if self.text is None:
            self.text = self.ax.text(0, 0, f"steps: {self.frame.step_}",
                                     fontsize=12, color='black')
        else:
            self.text.set_text(f"steps: {self.frame.step_}")

        # 绘制车辆
        for lane in self.lane_list:
            for car in lane.car_list:
                polygon = car.get_bbox()
                # 将车辆颜色从0-255范围转换为0-1范围
                normalized_color = self.normalize_color(car.color)
                car_patch = Polygon(polygon, facecolor=normalized_color, alpha=0.5,
                                    edgecolor='black', linewidth=1, closed=True)
                self.ax.add_patch(car_patch)

        plt.pause(0.01)  # 短暂暂停以更新显示
        if self.frame_rate > 0:
            plt.pause(1 / self.frame_rate)
