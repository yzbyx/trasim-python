# -*- coding: utf-8 -*-
# @time : 2023/6/3 19:35
# @Author : yzbyx
# @File : CFModel_CTM.py
# Software: PyCharm
from typing import Optional, TYPE_CHECKING

from trasim_simplified.core.constant import CFM
from trasim_simplified.core.kinematics.cfm import CFModel

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle


class CFModel_CTM(CFModel):
    def __init__(self, vehicle: Optional['Vehicle'], f_param: dict[str, float]):
        super().__init__(vehicle)
        self.name = CFM.CTM
        self.thesis = ""

        self._v0 = f_param.get("v0", 30.)
        """自由流流速 [m/s]"""
        self._wb = f_param.get("wb", 15. / 3.6)
        """后向传播波速 [m/s]"""
        self._qm = f_param.get("qm", 1800 / 3600)
        """最大通行能力 [veh/s]"""
        self._kj = f_param.get("kj", 160 / 1000)
        """拥挤密度 [veh/m]"""

        self.k1 = None
        self.k2 = None

    def get_jam_density(self, car_length):
        return self._kj

    def basic_diagram_k_to_q(self, dhw, car_length, speed_limit):
        self.cal_k1_k2()
        k = 1 / dhw
        assert k <= self._kj
        if k < self.k1:
            q = k * self._v0
        elif k >= self.k2:
            q = (self._kj - k) * self._wb
        else:
            q = self._qm
        return q

    def get_qm(self):
        return self._qm

    def cal_k1_k2(self):
        self.k1 = self._qm / self._v0
        self.k2 = self._kj - self._qm / self._wb
        if self.k1 > self.k2:
            self.k1 = self.k2 = self._wb * self._kj / (self._v0 + self._wb)

    def get_expect_dec(self):
        pass

    def get_expect_acc(self):
        pass

    def get_expect_speed(self):
        pass

    def _update_dynamic(self):
        pass

    def step(self, index, *args):
        pass
