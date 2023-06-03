# -*- coding: utf-8 -*-
# @Time : 2023/6/3 13:22
# @Author : yzbyx
# @File : ctm_road.py
# Software: PyCharm
from typing import Optional

from trasim_simplified.core.frame.macro.ctm_lane import CTM_Lane


class CTM_Road:
    def __init__(self):
        self.lane_list: Optional[list[CTM_Lane]] = None
