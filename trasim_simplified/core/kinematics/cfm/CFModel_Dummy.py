# -*- coding: utf-8 -*-
# @Time : 2023/5/15 17:39
# @Author : yzbyx
# @File : CFModel_Dummy.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.msg.trasimError import TrasimError

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle

from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.constant import CFM


class CFModel_Dummy(CFModel):
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)

    def get_expect_dec(self):
        return self.DEFAULT_EXPECT_DEC

    def get_expect_acc(self):
        return self.DEFAULT_EXPECT_ACC

    def get_expect_speed(self):
        return 0.

    def _update_dynamic(self):
        pass

    def step(self, index, *args):
        return 0
