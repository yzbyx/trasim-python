# -*- coding: utf-8 -*-
# @Time : 2023/5/12 16:22
# @Author : yzbyx
# @File : __init__.py.py
# Software: PyCharm
from typing import Optional

from trasim_simplified.core.constant import LCM
from trasim_simplified.core.kinematics.lcm.LCModel import LCModel
from trasim_simplified.core.kinematics.lcm.LCModel_ACC import LCModel_ACC
from trasim_simplified.core.kinematics.lcm.LCModel_KK import LCModel_KK
from trasim_simplified.msg.trasimError import ErrorMessage as rem, TrasimError

__All__ = ['get_lc_model', 'LCModel']


def get_lc_model(_driver, name=LCM.KK, param=None) -> Optional[LCModel]:
    if param is None:
        param = {}
    if name == LCM.KK:
        return LCModel_KK(_driver, param)
    if name == LCM.ACC:
        return LCModel_ACC(_driver, param)
    if name is None:
        return None
    else:
        raise TrasimError(rem.NO_MODEL.format(name))
