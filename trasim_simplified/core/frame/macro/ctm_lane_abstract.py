# -*- coding: utf-8 -*-
# @Time : 2023/5/24 12:20
# @Author : yzbyx
# @File : ctm_lane_abstract.py
# Software: PyCharm
from typing import Optional

import numpy as np

from trasim_simplified.core.frame.macro.ctm_road import CTM_Road
from trasim_simplified.core.kinematics.cfm import CFModel, get_cf_model


class CTM_Lane:
    def __init__(self, is_circle=False):
        self.ID = 0
        self.index = 0
        self.is_circle = is_circle
        self.road: Optional[CTM_Road] = None
        self.cell_speed_limit: Optional[list[float]] = None
        self.cell_length: Optional[list[float]] = None
        self.cell_diagram: Optional[list[CFModel]] = None
        self.cell_car_length: Optional[list[float]] = None
        self.cell_density: Optional[list[float]] = None
        self.cell_speed: Optional[list[float]] = None

        self.flow_in: Optional[float] = None
        self.flow_out: Optional[float] = None

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

    def cell_config(self, cell_length: float, cell_num: int, cfm_name: str, cfm_param: dict[str, float], car_length=5.,
                    speed_limit=30., initial_density=0., initial_speed=0.):
        self.cell_length.extend([cell_length] * cell_num)
        cfm = get_cf_model(None, cfm_name, cfm_param)
        self.cell_diagram.extend([cfm] * cell_num)
        self.cell_car_length.extend([car_length] * cell_num)
        self.cell_speed_limit.extend([speed_limit] * cell_num)
        self.cell_density.extend([initial_density] * cell_num)
        self.cell_speed.extend([initial_speed] * cell_num)

    def boundary_condition_config(self, flow_in=0, flow_out=np.inf):
        self.flow_in = flow_in
        self.flow_out = flow_out

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
        self.force_speed_limit = kwargs.get("force_speed_limit", False)
        """是否强制车辆速度不超过道路限速"""
        # TODO: UI
        if self.has_ui and not self.road_control:
            self.ui.ui_init(caption=caption, frame_rate=frame_rate)

        # 整个仿真能够运行sim_step的仿真步
        while self.sim_step != self.step_:
            if not self.is_circle:
                self.traffic_generation()
            # 能够记录warm_up_step仿真步时的车辆数据
            if self.data_save and self.step_ >= self.warm_up_step:
                self.record()
            self.step()  # 未更新状态，但已经计算跟驰结果
            # 控制车辆对应的step需要在下一个仿真步才能显现到数据记录中
            if self.yield_: yield self.step_
            self.update_state()  # 更新车辆状态
            if self.road_control: yield self.step_
            self.step_ += 1
            self.time_ += self.dt
            if self.has_ui and not self.road_control: self.ui.ui_update()
