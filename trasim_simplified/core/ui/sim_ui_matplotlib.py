# -*- coding = uft-8 -*-
# @time : 2023-03-31 16:02
# @Author : yzbyx
# @File : sim_ui_matplotlib.py
# @Software : PyCharm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
        self.static_item = []  # 存储道路相关的patches
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
            y_min = veh.y - 5
            y_max = veh.y + 5

            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)

    def resize_by_car(self):
        """
        根据车辆数量调整UI大小
        """
        if self.ax is not None:
            # 计算新的x轴范围
            x_min = min(car.x for lane in self.lane_list for car in lane.car_list) - 10
            x_max = max(car.x for lane in self.lane_list for car in lane.car_list) + 10

            # 设置x轴范围
            self.ax.set_xlim(x_min, x_max)
            # 设置y轴范围
            y_min = min(car.y for lane in self.lane_list for car in lane.car_list) - 5
            y_max = max(car.y for lane in self.lane_list for car in lane.car_list) + 5
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

        self.static_item.extend(self.frame.draw(self.ax))
        self.ui_update()  # 更新显示

    def ui_update(self):
        """更新UI显示，只更新动态内容"""
        # 清除之前的车辆patches
        for patch in self.ax.patches:
            if patch not in self.static_item:
                patch.remove()
        for line in self.ax.lines:
            if line not in self.static_item:
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
                car_type = "AV" if hasattr(car, "mpc_solver") else "HV"
                text = self.ax.text(
                    car.x + car.length / 2, car.y, f"{car.ID} {car_type}", fontsize=12,
                    color='white', ha='center', va='center'
                )
                self.text_list.append(text)
                # 如果换道，且为AV，则绘制换道轨迹
                if hasattr(car, "mpc_solver") and car.mpc_solver is not None:
                    lc_traj = car.mpc_solver.ref_path.ref_path
                    self.ax.plot(lc_traj[:, 0], lc_traj[:, 3],
                                 color='blue', linestyle='--', linewidth=1)

        self.resize_by_car()

        plt.pause(0.01)  # 短暂暂停以更新显示
        if self.frame_rate > 0:
            plt.pause(1 / self.frame_rate)
