# -*- coding: utf-8 -*-
# @time : 2023/5/24 12:20
# @Author : yzbyx
# @File : ctm_lane.py
# Software: PyCharm
from typing import Optional, TYPE_CHECKING

import numpy as np

from trasim_simplified.core.kinematics.cfm import CFModel, get_cf_model
from trasim_simplified.core.ui.ctm_ui import CTM_UI

if TYPE_CHECKING:
    from trasim_simplified.core.frame.macro.ctm_road import CTM_Road


class CTM_Lane:
    def __init__(self, is_circle=False):
        self.ID = 0
        self.index = 0
        self.is_circle = is_circle
        self.road: Optional[CTM_Road] = None
        self.cell_speed_limit: Optional[list[float]] = []
        self.cell_length: Optional[list[float]] = []
        self.cell_diagram: Optional[list[CFModel]] = []

        self.cell_car_length: Optional[list[float]] = []
        """车辆长度 [m]"""
        self.cell_car_num: Optional[list[float]] = []
        self.cell_speed: Optional[list[float]] = []
        """交通流流速 [m/s]"""
        self.cell_density: Optional[list[float]] = []
        """交通流密度 [veh/m]"""
        self.cell_flow: Optional[list[float]] = []
        """交通流流量 [veh/s]"""

        self.cell_flow_in: Optional[list[float]] = []
        self.cell_flow_out: Optional[list[float]] = []
        self.cell_jam_density: Optional[list[float]] = []
        self.cell_car_length_upstream: Optional[list[float]] = []

        self.flow_in: Optional[float] = None
        """边界流入流量 [veh/s]"""
        self.flow_out: Optional[float] = None
        """边界最大流出流量 [veh/s]"""
        self.flow_in_car_length = None

        self.cell_speed_list = []
        self.cell_density_list = []
        self.cell_flow_list = []
        self.cell_flow_in_list = []
        self.cell_flow_out_list = []

        self.step_list = []
        self.time_list = []

        self.lane_length = 0
        self.cell_start_pos = []

        self.step_ = 0
        """当前仿真步次"""
        self.time_ = 0
        """当前仿真时长 [s]"""
        self.yield_ = True
        """run()是否为迭代器"""
        self.road_control = False
        """是否为Road类控制"""
        self.force_speed_limit = False
        """是否强制车辆速度不超过道路限速"""
        self.dt = 0.1
        """仿真步长 [s]"""
        self.warm_up_step = int(5 * 60 / self.dt)
        """预热时间 [s]"""
        self.sim_step = int(10 * 60 / self.dt)
        """总仿真步 [次]"""
        self.data_save = False
        """是否保存数据"""
        self.has_ui = False
        self.ui = CTM_UI(self)

    def cell_config(self, cell_length: float, cell_num: int, cfm_name: str, cfm_param: dict[str, float], car_length=5.,
                    speed_limit=30., initial_density=0):
        self.cell_length.extend([cell_length] * cell_num)
        cfm = get_cf_model(None, cfm_name, cfm_param)
        self.cell_diagram.extend([cfm] * cell_num)
        self.cell_car_length.extend([car_length] * cell_num)
        self.cell_speed_limit.extend([speed_limit] * cell_num)
        self.cell_car_num.extend([initial_density * cell_length] * cell_num)
        density = initial_density
        self.cell_density.extend([density] * cell_num)
        flow = self.cell_diagram[-1].basic_diagram_k_to_q(1 / initial_density, car_length, speed_limit)
        jam_density = self.cell_diagram[-1].get_jam_density(car_length)
        self.cell_jam_density.extend([jam_density] * cell_num)
        self.cell_flow.extend([flow] * cell_num)
        self.cell_speed.extend([flow / density] * cell_num)

        for i in range(cell_num):
            self.cell_start_pos.append(self.lane_length + cell_length * i)
        self.lane_length += cell_length * cell_num

    def boundary_condition_config(self, flow_in_car_length, flow_in=0, flow_out=np.inf):
        self.flow_in_car_length = flow_in_car_length
        self.flow_in = flow_in / 3600
        self.flow_out = flow_out / 3600

    def run(self, data_save=True, has_ui=True, **kwargs):
        if kwargs is None:
            kwargs = {}
        self.data_save = data_save
        """是否记录数据"""
        self.warm_up_step = kwargs.get("warm_up_step", int(5 * 60 / self.dt))
        """预热步数 [s]"""
        self.dt = kwargs.get("dt", 1)
        """仿真步长 [s]"""
        self.sim_step = kwargs.get("sim_step", int(10 * 60 / self.dt))
        """总仿真步 [次]"""
        frame_rate = kwargs.get("frame_rate", -1)
        """pygame刷新率 [fps]"""
        caption = kwargs.get("ui_caption", "微观交通流仿真")
        self.yield_ = kwargs.get("if_yield", True)
        """run()是否为迭代器"""
        self.has_ui = has_ui
        """是否显示UI"""
        if self.has_ui and not self.road_control:
            self.ui.ui_init(caption=caption, frame_rate=frame_rate)

        # 整个仿真能够运行sim_step的仿真步
        while self.sim_step != self.step_:
            # 能够记录warm_up_step仿真步时的车辆数据
            if self.data_save and self.step_ >= self.warm_up_step:
                self.record()
            self.step()  # 未更新状态，但已经计算每个元胞的流入流量
            if self.yield_: yield self.step_
            self.update_state()  # 更新元胞状态
            if self.road_control: yield self.step_
            self.step_ += 1
            self.time_ += self.dt
            if self.has_ui and not self.road_control: self.ui.ui_update()

    def record(self):
        self.step_list.append([self.step_] * len(self.cell_flow))
        self.time_list.append([self.time_] * len(self.cell_flow))
        self.cell_flow_list.append(self.cell_flow.copy())
        self.cell_density_list.append(self.cell_density.copy())
        self.cell_speed_list.append(self.cell_speed.copy())
        self.cell_flow_in_list.append(self.cell_flow_in.copy())
        self.cell_flow_out_list.append(self.cell_flow_out.copy())

    def step(self):
        self.cell_flow_in = []
        for i, density in enumerate(self.cell_density):
            if i != 0:
                flow_in = min(
                    self.cell_diagram[i - 1].basic_diagram_k_to_q(
                        1 / self.cell_density[i - 1], self.cell_car_length[i - 1], self.cell_speed_limit[i - 1]
                    ), self.cell_density[i - 1] * self.cell_length[i - 1] / self.dt,
                    (self.cell_jam_density[i] - self.cell_density[i]) * self.cell_length[i] / self.dt)
            else:
                if self.is_circle:
                    flow_in = min(
                        self.cell_diagram[-1].basic_diagram_k_to_q(
                            1 / self.cell_density[-1], self.cell_car_length[-1], self.cell_speed_limit[-1]
                        ), self.cell_density[-1] * self.cell_length[-1] / self.dt,
                        (self.cell_jam_density[i] - self.cell_density[i]) * self.cell_length[i] / self.dt)
                else:
                    flow_in = min(
                        self.flow_in,
                        (self.cell_jam_density[i] - self.cell_density[i]) * self.cell_length[i] / self.dt)
            self.cell_flow_in.append(flow_in)

        if not self.is_circle:
            self.cell_flow_out = self.cell_flow_in[1:]  # 此处缺失的在update_state函数里补充
            self.cell_car_length_upstream = [self.flow_in_car_length] + self.cell_car_length[1:]
        else:
            self.cell_flow_out = self.cell_flow_in[1:] + [self.cell_flow_in[0]]
            self.cell_car_length_upstream = [self.cell_car_length[-1]] + self.cell_car_length[:-1]

    def update_state(self):
        for i, flow_in in enumerate(self.cell_flow_in):
            if self.is_circle or i != len(self.cell_flow_in) - 1:
                # 更新平均车长
                flow_in_num = flow_in * self.dt
                stay_num = self.cell_car_num[i] - self.cell_flow_out[i] * self.dt
                self.cell_car_length[i] = (
                        flow_in_num * self.cell_car_length_upstream[i] +
                        stay_num * self.cell_car_length[i]
                ) / (flow_in_num + stay_num)
                self.cell_density[i] += (flow_in - self.cell_flow_out[i]) * self.dt / self.cell_length[i]
            else:
                flow_out = min(self.cell_diagram[-1].basic_diagram_k_to_q(
                    1 / self.cell_density[-1], self.cell_car_length[-1], self.cell_speed_limit[-1]
                ), self.cell_density[-1] * self.cell_length[-1] / self.dt, self.flow_out)
                self.cell_density[i] += (flow_in - flow_out) * self.dt / self.cell_length[-1]
                self.cell_flow_out.append(flow_out)  # 此处补充开边界末尾的flow_out
                # 更新平均车长
                flow_in_num = flow_in * self.dt
                stay_num = self.cell_car_num[i] - self.cell_flow_out[i] * self.dt
                self.cell_car_length[i] = (
                      flow_in_num * self.cell_car_length_upstream[i] +
                      stay_num * self.cell_car_length[i]
                ) / (flow_in_num + stay_num)

            self.cell_car_num[i] = self.cell_density[i] * self.cell_length[i]
            self.cell_flow[i] = self.cell_diagram[i].basic_diagram_k_to_q(
                1 / self.cell_density[i], self.cell_car_length[i], self.cell_speed_limit[i]
            )
            self.cell_speed[i] = self.cell_flow[i] / self.cell_density[i]

    @property
    def cell_occ(self):
        occ = []
        for i, car_length in enumerate(self.cell_car_length):
            occ.append(car_length * self.cell_car_num[i] / self.cell_length[i])
        return occ
