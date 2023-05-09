# -*- coding = uft-8 -*-
# @Time : 2022-04-27 21:31
# @Author : yzbyx
# @File : __init__.py
# @Software : PyCharm
from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.kinematics.cfm.CFModel_IDM import CFModel_IDM as IDM
from trasim_simplified.core.kinematics.cfm.CFModel_Gipps import CFModel_Gipps as GIPPS
from trasim_simplified.core.kinematics.cfm.CFModel_W99 import CFModel_W99 as W99
from trasim_simplified.core.kinematics.cfm.CFModel_NonLinearGHR import CFModel_NonLinearGHR as GHR
from trasim_simplified.core.kinematics.cfm.CFModel_OVM import CFModel_OVM as OVM
from trasim_simplified.core.kinematics.cfm.CFModel_KK import CFModel_KK as KK
from trasim_simplified.core.kinematics.cfm.CFModel_Linear import CFModel_Linear as Linear
from trasim_simplified.core.constant import CFM

from trasim_simplified.msg.trasimError import ErrorMessage as rem, TrasimError

__All__ = ['get_cf_model', 'CFModel']


def get_cf_model(_driver, name=CFM.IDM, param=None) -> CFModel:
    if param is None:
        param = {}
    if name == CFM.IDM:
        return IDM(_driver, param)
    elif name == CFM.GIPPS:
        return GIPPS(_driver, param)
    elif name == CFM.LINEAR:
        return Linear(_driver, param)
    elif name == CFM.WIEDEMANN_99:
        return W99(_driver, param)
    elif name == CFM.NON_LINEAR_GHR:
        return GHR(_driver, param)
    elif name == CFM.OPTIMAL_VELOCITY:
        return OVM(_driver, param)
    elif name == CFM.KK:
        return KK(_driver, param)
    else:
        raise TrasimError(rem.NO_MODEL.format(name))
