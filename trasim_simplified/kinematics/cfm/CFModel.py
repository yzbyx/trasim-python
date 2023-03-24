# -*- coding = uft-8 -*-
# @Time : 2022-04-04 10:56
# @Author : yzbyx
# @File : CFModel.py
# @Software : PyCharm
import random
from abc import ABC
from typing import TYPE_CHECKING, Optional

from trasim_simplified.core.constant import RUNMODE, RANDOM_SEED
from trasim_simplified.kinematics.model import Model

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

    def getTau(self) -> float:
        """获取反应时间"""
        return self.get_param_map().get('tau', 0)
