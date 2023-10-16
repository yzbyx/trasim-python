# -*- coding = uft-8 -*-
# @Time : 2022-04-27 21:31
# @Author : yzbyx
# @File : __init__.py
# @Software : PyCharm
from trasim_simplified.core.kinematics.cfm.CFModel import CFModel
from trasim_simplified.core.kinematics.cfm.CFModel_IDM import CFModel_IDM as IDM, cf_IDM_acc
from trasim_simplified.core.kinematics.cfm.CFModel_Gipps import CFModel_Gipps as GIPPS
from trasim_simplified.core.kinematics.cfm.CFModel_W99 import CFModel_W99 as W99
from trasim_simplified.core.kinematics.cfm.CFModel_NonLinearGHR import CFModel_NonLinearGHR as GHR
from trasim_simplified.core.kinematics.cfm.CFModel_OVM import CFModel_OVM as OVM
from trasim_simplified.core.kinematics.cfm.CFModel_KK import CFModel_KK as KK
from trasim_simplified.core.kinematics.cfm.CFModel_Linear import CFModel_Linear as Linear
from trasim_simplified.core.kinematics.cfm.CFModel_ACC import CFModel_ACC as ACC
from trasim_simplified.core.kinematics.cfm.CFModel_CACC import CFModel_CACC as CACC
from trasim_simplified.core.kinematics.cfm.CFModel_TPACC import CFModel_TPACC as TPACC
from trasim_simplified.core.kinematics.cfm.CFModel_LCM import CFModel_LCM as LCM
from trasim_simplified.core.kinematics.cfm.CFModel_CTM import CFModel_CTM as CTM
from trasim_simplified.core.kinematics.cfm.CFModel_Dummy import CFModel_Dummy as Dummy
from trasim_simplified.core.constant import CFM

from trasim_simplified.msg.trasimError import ErrorMessage as rem, TrasimError

__All__ = ['get_cf_model', 'CFModel', 'get_cf_id']


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
    elif name == CFM.ACC:
        return ACC(_driver, param)
    elif name == CFM.TPACC:
        return TPACC(_driver, param)
    elif name == CFM.DUMMY:
        return Dummy(_driver, param)
    elif name == CFM.CACC:
        return CACC(_driver, param)
    elif name == CFM.LCM:
        return LCM(_driver, param)
    elif name == CFM.CTM:
        return CTM(_driver, param)
    else:
        raise TrasimError(rem.NO_MODEL.format(name))


def get_cf_func(cf_name):
    if cf_name == CFM.IDM:
        from trasim_simplified.core.kinematics.cfm.CFModel_IDM import cf_IDM_acc
        cf_func = cf_IDM_acc
    elif cf_name == CFM.GIPPS:
        from trasim_simplified.core.kinematics.cfm.CFModel_Gipps import cf_Gipps_acc_jit
        cf_func = cf_Gipps_acc_jit
    elif cf_name == CFM.NON_LINEAR_GHR:
        from trasim_simplified.core.kinematics.cfm.CFModel_NonLinearGHR import cf_NonLinearGHR_acc
        cf_func = cf_NonLinearGHR_acc
    elif cf_name == CFM.WIEDEMANN_99:
        from trasim_simplified.core.kinematics.cfm.CFModel_W99 import cf_Wiedemann99_acc
        cf_func = cf_Wiedemann99_acc
    elif cf_name == CFM.OPTIMAL_VELOCITY:
        from trasim_simplified.core.kinematics.cfm.CFModel_OVM import cf_OVM_acc
        cf_func = cf_OVM_acc
    else:
        raise TrasimError(f"{cf_name} is not be configured!")
    return cf_func


def get_cf_id(name) -> int:
    if name == CFM.IDM:
        return 0
    elif name == CFM.GIPPS:
        return 1
    elif name == CFM.LINEAR:
        return 2
    elif name == CFM.WIEDEMANN_99:
        return 3
    elif name == CFM.NON_LINEAR_GHR:
        return 4
    elif name == CFM.OPTIMAL_VELOCITY:
        return 5
    elif name == CFM.KK:
        return 6
    elif name == CFM.ACC:
        return 7
    elif name == CFM.TPACC:
        return 8
    elif name == CFM.DUMMY:
        return -1
    elif name == CFM.CACC:
        return 9
    elif name == CFM.LCM:
        return 10
    elif name == CFM.CTM:
        return 11
    else:
        raise TrasimError(rem.NO_MODEL.format(name))
