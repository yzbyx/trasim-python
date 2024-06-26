# -*- coding = uft-8 -*-
# @Time : 2022-04-04 10:56
# @Author : yzbyx
# @File : CFModel.py
# @Software : PyCharm
import abc
import random
from abc import ABC
from typing import TYPE_CHECKING, Optional

import numpy as np

from trasim_simplified.core.constant import RUNMODE, RANDOM_SEED
from trasim_simplified.core.kinematics.model import Model

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle


class CFModel(Model, ABC):
    """车辆跟驰模型的抽象基类"""

    _RANDOM = random.Random(RANDOM_SEED.CFM_SEED)

    def __init__(self, vehicle: Optional['Vehicle']):
        super().__init__(vehicle)
        self.status = None
        self.mode = RUNMODE.NORMAL
        self.random = CFModel._RANDOM
        self.DEFAULT_EXPECT_DEC = 3.
        self.DEFAULT_EXPECT_ACC = 3.
        self.DEFAULT_EXPECT_SPEED = 30.

    @abc.abstractmethod
    def get_expect_dec(self):
        """值为正数"""
        pass

    @abc.abstractmethod
    def get_expect_acc(self):
        pass

    @abc.abstractmethod
    def get_expect_speed(self):
        pass

    def get_speed_limit(self):
        if self.vehicle.lane.force_speed_limit is False:
            return np.Inf
        return self.vehicle.lane.get_speed_limit(self.vehicle.x, self.vehicle.type)

    def equilibrium_state(self, *args):
        pass

    def basic_diagram_k_to_q(self, dhw, car_length, speed_limit):
        """veh/s"""
        pass

    def get_jam_density(self, car_length):
        """veh/m"""
        pass

    def get_qm(self):
        """veh/s"""
        pass
