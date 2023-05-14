# -*- coding: utf-8 -*-
# @Time : 2023/5/12 16:22
# @Author : yzbyx
# @File : LCModel.py
# Software: PyCharm
import abc
import random
from abc import ABC
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.constant import RUNMODE, RANDOM_SEED
from trasim_simplified.core.kinematics.model import Model

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle


class LCModel(Model, ABC):
    """车辆跟驰模型的抽象基类"""

    _RANDOM = random.Random(RANDOM_SEED.LCM_SEED)

    def __init__(self, vehicle: Optional['Vehicle']):
        super().__init__(vehicle)
        self.status = None
        self.mode = RUNMODE.NORMAL
        self.random = LCModel._RANDOM
        self.last_lc_time_ = - np.Inf
