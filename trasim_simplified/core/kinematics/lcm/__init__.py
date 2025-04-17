# -*- coding: utf-8 -*-
# @time : 2023/5/12 16:22
# @Author : yzbyx
# @File : __init__.py.py
# Software: PyCharm
from typing import Optional

from trasim_simplified.core.constant import LCM
from trasim_simplified.core.kinematics.lcm.LCModel import LCModel
from trasim_simplified.core.kinematics.lcm.LCModel_ACC import LCModel_ACC
from trasim_simplified.core.kinematics.lcm.LCModel_KK import LCModel_KK
from trasim_simplified.core.kinematics.lcm.LCModel_Mobil import LCModel_Mobil
from trasim_simplified.msg.trasimError import ErrorMessage as rem, TrasimError

__All__ = ['get_lc_model', 'LCModel', 'get_lc_id']


def get_lc_model(name=LCM.KK) -> [LCModel]:
    if name == LCM.KK:
        return LCModel_KK
    if name == LCM.ACC:
        return LCModel_ACC
    if name == LCM.MOBIL:
        return LCModel_Mobil
    else:
        raise TrasimError(rem.NO_MODEL.format(name))


def get_lc_id(name) -> int:
    if name == LCM.KK:
        return 0
    elif name == LCM.ACC:
        return 1
    elif name is None:
        return -1
    else:
        raise TrasimError(rem.NO_MODEL.format(name))
