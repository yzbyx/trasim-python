# -*- coding: utf-8 -*-
# @Time : 2023/5/24 12:20
# @Author : yzbyx
# @File : ctm_road.py
# Software: PyCharm
from typing import Optional

from trasim_simplified.core.kinematics.cfm import CFModel, get_cf_model


class CTM_Road:
    def __init__(self, lane_num: int):
        self.length = 0
        self.lane_num = lane_num
        self.cell_length: Optional[list[float]] = None
        self.cell_diagram: Optional[list[CFModel]] = None
        self.cell_car_length: Optional[list[float]] = None

    def cell_config(self, cell_length: float, cell_num: int, cfm_name: str, cfm_param: dict[str, float], car_length=5):
        self.cell_length.extend([cell_length] * cell_num)
        cfm = get_cf_model(None, cfm_name, cfm_param)
        self.cell_diagram.extend([cfm] * cell_num)
        self.length += cell_length * cell_num
        self.cell_car_length.extend([car_length] * cell_num)

    def run(self, dt: float):
        pass
