# -*- coding: utf-8 -*-
# @time : 2023/10/15 11:41
# @Author : yzbyx
# @File : run_quiver_module.py
# Software: PyCharm
from trasim_simplified.core.kinematics.cfm.CFModel_IDM import cf_IDM_acc, cf_IDM_equilibrium
from trasim_simplified.core.kinematics.cfm.CFM_IDM_SZ import cf_IDM_SZ_acc, cf_IDM_SZ_equilibrium
from trasim_simplified.core.kinematics.cfm.CFM_IDM_VS import cf_IDM_VS_acc, cf_IDM_VS_equilibrium
from trasim_simplified.util.interaction.quiver import QuiverInteract, cf_Zhang_acc, cf_Zhang_equilibrium

if __name__ == '__main__':
    q_IDM = QuiverInteract(
        cf_func=cf_IDM_acc,
        cf_e_func=cf_IDM_equilibrium,
        default={"s0": 2, "s1": 0, "v0": 33.3, "T": 1.6, "omega": 0.73, "d": 1.67, "delta": 4},
        range={"s0": [0, 5], "s1": [0, 5], "v0": [0, 40], "T": [0, 5], "omega": [0, 5], "d": [0, 10], "delta": [0, 10]},
        step={"s0": 0.1, "s1": 0.1, "v0": 0.1, "T": 0.1, "omega": 0.1, "d": 0.1, "delta": 1}
    ).run()

    # QuiverInteract(
    #     cf_func=cf_IDM_SZ_acc,
    #     cf_e_func=cf_IDM_SZ_equilibrium,
    #     default={"s0": 2, "T": 1.6, "omega": 0.73, "d": 1.67},
    #     range={"s0": [0, 5], "T": [0, 5], "omega": [0, 5], "d": [0, 5]},
    #     step={"s0": 0.1, "T": 0.1, "omega": 0.1, "d": 0.1}
    # ).run()

    # QuiverInteract(
    #     cf_func=cf_IDM_VS_acc,
    #     cf_e_func=cf_IDM_VS_equilibrium,
    #     default={"s0": 2, "s1": 0, "v0": 33.3, "T": 1.6, "omega": 0.73, "delta": 4},
    #     range={"s0": [0, 5], "s1": [0, 5], "v0": [0, 40], "T": [0, 5], "omega": [0, 5], "delta": [0, 10]},
    #     step={"s0": 0.1, "s1": 0.1, "v0": 0.1, "T": 0.1, "omega": 0.1, "delta": 1}
    # ).run()

    # QuiverInteract(
    #     cf_func=cf_Zhang_acc,
    #     cf_e_func=cf_Zhang_equilibrium,
    #     default={"alpha": 0.5, "beta": 0.5, "v0": 30, "s0": 2, "T": 1.6},
    #     range={"alpha": [0, 10], "beta": [0, 10], "v0": [0, 40], "s0": [0, 5], "T": [0, 5]},
    #     step={"alpha": 0.1, "beta": 0.1, "v0": 0.1, "s0": 0.1, "T": 0.1}
    # ).run()


