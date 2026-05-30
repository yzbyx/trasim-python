# -*- coding: utf-8 -*-
# @time : 2023/6/3 17:28
# @Author : yzbyx
# @File : run_ctm_lane.py
# Software: PyCharm
from trasim_simplified.core.constant import CFM
from trasim_simplified.core.frame.macro.ctm_lane import CTM_Lane
from trasim_simplified.util.timer import timer_no_log


@timer_no_log
def run_ctm_lane():
    _cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True}
    dt = 1
    warm_up_step = 0
    sim_step = warm_up_step + int(300 / dt)

    is_circle = True

    sim = CTM_Lane(is_circle)
    if is_circle:
        sim.cell_config(50, 18, CFM.IDM, _cf_param, 5, speed_limit=30., initial_density=30 / 1000)
        sim.cell_config(50, 2, CFM.IDM, _cf_param, 5, speed_limit=30., initial_density=10 / 1000)
    else:
        sim.cell_config(50, 20, CFM.IDM, _cf_param, 5, speed_limit=30., initial_density=50 / 1000)
        sim.boundary_condition_config(5, flow_in=2000)

    for step in sim.run(data_save=True, has_ui=True, frame_rate=10,
                        warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        print(step)
        print(f"sim.cell_occ: {sim.cell_occ}")
        print(f"sim.cell_density: {sim.cell_density}")
        print(f"sim.cell_flow_in: {sim.cell_flow_in}")
        print(f"sim.cell_speed: {sim.cell_speed}")


if __name__ == '__main__':
    run_ctm_lane()
