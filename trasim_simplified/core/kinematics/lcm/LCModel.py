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
    from trasim_simplified.core.agent.vehicle import Vehicle


class LCModel(Model, ABC):
    """车辆跟驰模型的抽象基类"""

    _RANDOM = random.Random(RANDOM_SEED.LCM_SEED)

    def __init__(self, vehicle: Optional['Vehicle']):
        super().__init__(vehicle)
        self.status = None
        self.mode = RUNMODE.NORMAL
        self.random = LCModel._RANDOM
        self.last_lc_time_ = - np.inf

    @abc.abstractmethod
    def base_cal(self):
        """基本路段的普通换道"""
        pass

    @abc.abstractmethod
    def on_ramp_cal(self):
        """强制换道（例如匝道汇入）"""
        pass
