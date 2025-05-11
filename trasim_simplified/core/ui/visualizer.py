import numpy as np
import matplotlib
from typing import TYPE_CHECKING, Optional
from matplotlib.backend_bases import KeyEvent
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from trasim_simplified.core.agent.vehicle import Vehicle

matplotlib.use('TkAgg')


if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.road import Road


class Visualizer:
    def __init__(self, road: "Road" = None):
        self.fig_width = 9  # 画布宽度
        self.fig_height = 6  # 画布高度
        self.fig: Optional[plt.Figure] = None  # 画布实例
        self.vis_distance = 40  # 测试详情区域主车周围可视化范围
        self.object_color = {  # 不同类型物体的颜色映射
            'ego': 'orange',
            'vehicle': 'cornflowerblue',
            'bicycle': 'lightgreen',
            'pedestrian': 'lightcoral',
        }
        self.road = road

        self.ego: Vehicle = road.choose_vehicle()  # 控制的车辆
        self.focus_mode = True if self.ego else False  # 是否聚焦在主车上
        self.global_ax_range = None  # 全局坐标轴范围

        self._replay_create_ax()
        self._plot_static()
        self.update_dynamic()
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        plt.show()

    def key_press_event(self, event: KeyEvent):
        if event.key == 'right':
            f, r = self.road.preceding_follower_vehicles(self.controlled_vehicle, self.controlled_vehicle.lane_index)
            if f:
                self.controlled_vehicle = f
        elif event.key == 'left':
            f, r = self.road.preceding_follower_vehicles(self.controlled_vehicle, self.controlled_vehicle.lane_index)
            if r:
                self.controlled_vehicle = r
        elif event.key == 'up':
            side_lanes = self.road.network.my_side_lanes(self.controlled_vehicle.lane_index)
            if side_lanes[0]:
                lane_index = side_lanes[0]
                f, r = self.road.preceding_follower_vehicles(self.controlled_vehicle, lane_index)
                if r:
                    self.controlled_vehicle = r
                elif f:
                    self.controlled_vehicle = f
        elif event.key == 'down':
            side_lanes = self.road.network.my_side_lanes(self.controlled_vehicle.lane_index)
            if side_lanes[1]:
                lane_index = side_lanes[1]
                f, r = self.road.preceding_follower_vehicles(self.controlled_vehicle, lane_index)
                if r:
                    self.controlled_vehicle = r
                elif f:
                    self.controlled_vehicle = f

        if self.controlled_vehicle:
            if isinstance(self.controlled_vehicle, IDMVehicle):
                if event.key == "w":
                    self.controlled_vehicle.TIME_WANTED = max(1, self.controlled_vehicle.TIME_WANTED - 0.5)
                elif event.key == "s":
                    self.controlled_vehicle.TIME_WANTED = min(5, self.controlled_vehicle.TIME_WANTED + 0.5)

    def _replay_create_ax(self):
        """进行可视化回放的视图创建和划分"""
        # 创建画布
        self.fig_width = 9
        self.fig_height = 6
        self.fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        # 划分网格
        gs = GridSpec(3, 2, width_ratios=[2, 5], height_ratios=[10, 5, 5])
        self.ax_table: plt.Axes = plt.subplot(gs[:2, 0])  # 左上角表格区域
        self.ax_map_bg: plt.Axes = plt.subplot(gs[2, 0])  # 左下角地图区域
        self.ax_detail_bg: plt.Axes = plt.subplot(gs[:, 1])  # 右边测试详情区域
        # 创建动态元素图层
        self.ax_detail_obj: plt.Axes = self.ax_detail_bg.twinx()
        self.ax_map_obj: plt.Axes = self.ax_map_bg.twinx()
        # 网格区域初始设置
        self.ax_map_bg.set_xticks([])
        self.ax_map_bg.set_yticks([])
        self.ax_detail_obj.set_yticks([])
        self.ax_map_obj.set_yticks([])
        self.ax_table.axis('off')
        plt.ion()

    def _plot_static(self):
        """进行可视化回放的静态信息绘制"""
        # 绘制左上角表格区域
        self.table = self._plot_table(self.ax_table)
        # 绘制右边测试详情区域
        self._plot_roads(self.ax_detail_bg)

        if self.focus_mode:
            ax_detail_range = self._update_ax_limit(
                self.ax_detail_bg,
                [self.ego.position[0] - self.vis_distance,
                 self.ego.position[0] + self.vis_distance],
                [self.ego.position[1] - self.vis_distance,
                 self.ego.position[1] + self.vis_distance]
            )
        else:
            min_x, max_x, min_y, max_y = self.road.get_road_boundaries()
            ax_detail_range = self._update_ax_limit(
                self.ax_detail_bg,
                [min_x, max_x],
                [min_y, max_y]
            )
            self.global_ax_range = ax_detail_range
        self.ax_detail_obj.set_ylim(ax_detail_range[1][0], ax_detail_range[1][1])

        # 绘制左下角地图区域
        self._plot_roads(self.ax_map_bg)
        ax_map_range = self._update_ax_limit(self.ax_map_bg, *self._get_road_boundary())
        self.ax_map_obj.set_ylim(ax_map_range[1][0], ax_map_range[1][1])
        self.position_box: patches.Rectangle = self._plot_position_box(self.ax_map_obj, ax_detail_range)
        plt.tight_layout()

    def update_dynamic(self):
        """对每一帧进行画布更新"""
        self.ax_detail_obj.cla()
        self.ax_detail_obj.set_yticks([])

        if self.focus_mode:
            # 更新表格信息
            self._update_table()
            # 更新右边测试详情区域物体信息
            new_range = self._update_ax_limit(
                self.ax_detail_bg,
                [self.ego.position[0] - self.vis_distance, self.ego.position[0] + self.vis_distance],
                [self.ego.position[1] - self.vis_distance, self.ego.position[1] + self.vis_distance]
            )
            self.ax_detail_obj.set_ylim(new_range[1][0], new_range[1][1])
            # 更新左下角地图区域定位框位置
            if new_range:
                self.position_box.set_xy([new_range[0][0], new_range[1][0]])
        else:
            self.ax_detail_obj.set_ylim(self.ax_detail_bg.get_ylim())
        self._plot_single_object(self.ax_detail_obj)
        self._plot_ego_info(self.ax_detail_obj) if self.focus_mode else None
        plt.pause(1e-7)

    def _plot_roads(self, ax: plt.Axes, draw_arrow: bool = False) -> None:
        """根据parse_opendrive模块解析出的opendrive路网信息绘制道路"""
        xlim1 = float("Inf")
        xlim2 = -float("Inf")
        ylim1 = float("Inf")
        ylim2 = -float("Inf")
        color = "gray"
        label = None

        for discrete_lane in self.road.network.lanes_list():
            verts = []
            codes = [Path.MOVETO]

            for x, y in np.vstack(
                    [discrete_lane.left_vertices, discrete_lane.right_vertices[::-1]]
            ):
                verts.append([x, y])
                codes.append(Path.LINETO)

                # if color != 'gray':
                xlim1 = min(xlim1, x)
                xlim2 = max(xlim2, x)

                ylim1 = min(ylim1, y)
                ylim2 = max(ylim2, y)

            verts.append(verts[0])
            codes[-1] = Path.CLOSEPOLY

            path = Path(verts, codes)

            ax.add_patch(
                patches.PathPatch(
                    path,
                    facecolor=color,
                    edgecolor="none",
                    lw=0.0,
                    alpha=0.5,
                    zorder=0,
                    label=label,
                )
            )

            if discrete_lane.line_types[0] != LineType.NONE:
                ax.plot(
                    [x for x, _ in discrete_lane.left_vertices],
                    [y for _, y in discrete_lane.left_vertices],
                    color="yellow" if discrete_lane.line_types[0] == LineType.CONTINUOUS_LINE else 'white',
                    lw=1,
                    zorder=1,
                    linestyle='dashed' if discrete_lane.line_types[0] == LineType.STRIPED else 'solid',
                )
            if discrete_lane.line_types[1] != LineType.NONE:
                ax.plot(
                    [x for x, _ in discrete_lane.right_vertices],
                    [y for _, y in discrete_lane.right_vertices],
                    color="yellow" if discrete_lane.line_types[1] == LineType.CONTINUOUS_LINE else 'white',
                    lw=1,
                    zorder=1,
                    linestyle='dashed' if discrete_lane.line_types[1] == LineType.STRIPED else 'solid',
                )

            ax.plot(
                [x for x, _ in discrete_lane.center_vertices],
                [y for _, y in discrete_lane.center_vertices],
                color="gray",
                alpha=0.5,
                lw=0.8,
                zorder=1,
            )

    def _plot_position_box(self, ax: plt.Axes, range):
        """绘制地图区域的定位框"""
        position_box = patches.Rectangle(
            xy=(range[0][0], range[1][0]),
            width=range[0][1] - range[0][0],
            height=range[1][1] - range[1][0],
            angle=0,
            color="blue",
            fill=True,
            alpha=0.4,
            zorder=5
        )
        ax.add_patch(position_box)
        return position_box

    def _plot_single_object(self, ax: plt.Axes, c='cornflowerblue'):
        """利用 matplotlib 和 patches 绘制小汽车，以 x 轴为行驶方向"""
        objs = self.road.objects
        vehicles = self.road.vehicles
        items = objs + vehicles
        for obj in items:
            x, y, yaw, width, length = obj.position[0], obj.position[1], obj.heading, obj.WIDTH, obj.LENGTH

            angle = np.arctan(width / length) + yaw
            diagonal = np.sqrt(length ** 2 + width ** 2)
            ax.add_patch(
                patches.Rectangle(
                    xy=(x - diagonal / 2 * np.cos(angle),
                        y - diagonal / 2 * np.sin(angle)),
                    width=length,
                    height=width,
                    angle=yaw / np.pi * 180,
                    # color=c,
                    facecolor=c,
                    fill=True,
                    zorder=4,
                    edgecolor='red' if self.focus_mode and obj.id_ == self.controlled_vehicle.id_ else 'black'
                ))
            ax.annotate(obj.id_, (x, y), fontsize=8, zorder=5)

    def _plot_ego_info(self, ax: plt.Axes):
        if isinstance(self.ego, MPC_Vehicle):
            if self.ego.ref_path is not None:
                ax.plot(self.ego.ref_path[:, 0], self.ego.ref_path[:, 3], 'r--', lw=1)
                index = self.ego.lc_entered_step - self.ego.lc_start_step
                ax.scatter(self.ego.ref_path[index, 0], self.ego.ref_path[index, 3], c='r', s=50, marker='x')
            if hasattr(self.ego, 'lc_direction'):
                self.table[(8, 1)].get_text().set_text(self.ego.lc_direction)
            if hasattr(self.ego, 'apfs'):
                if self.ego.apfs is not None:
                    x, y = self.ego.position[0], self.ego.position[1]
                    for i, apf in enumerate(self.ego.apfs):
                        ax.annotate(
                            f"apf: {apf:.2f}",
                            (x + self.ego.length, y + (- i + 1) * self.ego.lane.DEFAULT_WIDTH),
                            fontsize=8, color='red'
                        )
            if hasattr(self.ego, 'safe_gaps'):
                if self.ego.safe_gaps is not None:
                    x, y = self.ego.position[0], self.ego.position[1]
                    for i, gap in enumerate(self.ego.safe_gaps):
                        if i == 0 and gap is not None:
                            ax.plot([x, x + (gap + self.ego.LENGTH / 2)],
                                    [y + self.ego.lane.DEFAULT_WIDTH, y + self.ego.lane.DEFAULT_WIDTH], 'b--', lw=1)
                            ax.annotate(f"lf_safe_gap: {gap:.2f}",
                                        (x + (gap + self.ego.LENGTH / 2) / 2, y + self.ego.lane.DEFAULT_WIDTH - 1),
                                        fontsize=8, color='blue')
                        elif i == 1 and gap is not None:
                            ax.plot([x, x - (gap + self.ego.LENGTH / 2)],
                                    [y + self.ego.lane.DEFAULT_WIDTH, y + self.ego.lane.DEFAULT_WIDTH], 'b--', lw=1)
                            ax.annotate(f"lr_safe_gap: {gap:.2f}",
                                        (x - (gap + self.ego.LENGTH / 2) / 2, y + self.ego.lane.DEFAULT_WIDTH - 1),
                                        fontsize=8, color='blue')
                        elif i == 2 and gap is not None:
                            ax.plot([x, x + (gap + self.ego.LENGTH / 2)],
                                    [y - self.ego.lane.DEFAULT_WIDTH, y - self.ego.lane.DEFAULT_WIDTH], 'b--', lw=1)
                            ax.annotate(f"rf_safe_gap: {gap:.2f}",
                                        (x + (gap + self.ego.LENGTH / 2) / 2, y - self.ego.lane.DEFAULT_WIDTH - 1),
                                        fontsize=8, color='blue')
                        elif i == 3 and gap is not None:
                            ax.plot([x, x - (gap + self.ego.LENGTH / 2)],
                                    [y - self.ego.lane.DEFAULT_WIDTH, y - self.ego.lane.DEFAULT_WIDTH], 'b--', lw=1)
                            ax.annotate(f"rr_safe_gap: {gap:.2f}",
                                        (x - (gap + self.ego.LENGTH / 2) / 2, y - self.ego.lane.DEFAULT_WIDTH - 1),
                                        fontsize=8, color='blue')
        if isinstance(self.ego, Game_A_Vehicle):
            if self.ego.lc_acc_gain is not None:
                x, y = self.ego.position[0], self.ego.position[1]
                for i, acc_gain in enumerate(self.ego.lc_acc_gain):
                    ax.annotate(
                        f"acc_gain: {acc_gain:.2f}",
                        (x + self.ego.length, y + (- i * 2 + 1) * self.ego.lane.DEFAULT_WIDTH),
                        fontsize=8, color='red'
                    )

    def _plot_table(self, ax):
        """绘制初始化表格信息"""
        data = [['dt', None],
                ['t', None],
                ['acc', None],
                ['rot', None],
                ['x_ego', None],
                ['y_ego', None],
                ['v_ego', None],
                ['yaw_ego', None],
                ['lc_status', None],
                ["controlled_T", None]
                ]
        table = ax.table(cellText=data, loc='upper left', cellLoc='center', fontsize=10, colWidths=[0.3, 0.65])

        table.auto_set_font_size(False)
        table.auto_set_column_width([0])
        table.SCALE(1, 1.4)

        table[(0, 1)].set_fontsize(8)
        table[(2, 1)].set_text_props(fontweight='bold', color='red')
        table[(3, 1)].set_text_props(fontweight='bold', color='red')

        return table

    def _update_table(self) -> None:
        """更新表格中的动态信息"""
        test_info = {
            'dt': self._format_number(1 / self.env.config["simulation_frequency"]),
            't': self._format_number(self.env.time),
            'acc': self._format_number(self.ego.action["acceleration"]),
            'rot': self._format_number(self.ego.action["steering"]),
            'ego_x': self._format_number(self.ego.position[0]),
            'ego_y': self._format_number(self.ego.position[1]),
            'ego_v': self._format_number(self.ego.speed),
            'ego_yaw': self._format_number(self.ego.heading),
        }

        for i, val in enumerate(test_info.values()):
            self.table[(i, 1)].get_text().set_text(val)

        if isinstance(self.controlled_vehicle, IDMVehicle):
            self.table[(9, 1)].get_text().set_text(self._format_number(self.controlled_vehicle.TIME_WANTED))

    def _update_ax_limit(self, ax: plt.Axes, range_x, range_y):
        """更新子图的坐标轴范围"""
        aspect_ratio = self._cal_aspect_ratio(ax, self.fig_width, self.fig_height)
        new_range_x, new_range_y = self._cal_proportional_range(range_x, range_y, aspect_ratio)
        ax.set_xlim(new_range_x[0], new_range_x[1])
        ax.set_ylim(new_range_y[0], new_range_y[1])
        return [new_range_x, new_range_y]

    @staticmethod
    def _format_number(number: float, digit: int = 3) -> str:
        if number is not None:
            return f"{number:.{digit}f}"
        return ""

    @staticmethod
    def _cal_proportional_range(range_x, range_y, aspect_ratio):
        """计算比例范围
        Args:
            range_x (list): x轴范围
            range_y (list): y轴范围
            aspect_ratio (float): 长宽比
        Returns:
            list: 比例范围
        """
        len_x = range_x[1] - range_x[0]
        len_y = range_y[1] - range_y[0]
        center = [(range_x[0] + range_x[1]) / 2, (range_y[0] + range_y[1]) / 2]

        if len_x > len_y * aspect_ratio:
            len_y = len_x / aspect_ratio
        else:
            len_x = len_y * aspect_ratio
        return [center[0] - len_x / 2, center[0] + len_x / 2], [center[1] - len_y / 2, center[1] + len_y / 2]

    @staticmethod
    def _cal_aspect_ratio(ax: plt.Axes, fig_width, fig_height):
        """计算子图的宽高比"""
        width = ax.get_position().width * fig_width
        height = ax.get_position().height * fig_height
        aspect_ratio = width / height
        return aspect_ratio

    def _get_road_boundary(self):
        xs, ys = [], []
        for lane in self.road.network.lanes_list():
            begin = lane.position(0, 0)
            end = lane.position(lane.length, 0)
            xs.extend([begin[0], end[0]])
            ys.extend([begin[1], end[1]])
        return [np.min(xs), np.max(xs)], [np.min(ys), np.max(ys)]
