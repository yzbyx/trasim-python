# -*- coding = uft-8 -*-
# @Time : 2023-04-09 20:20
# @Author : yzbyx
# @File : tests_frame.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.core.constant import CFM, V_TYPE
from trasim_simplified.core.frame.circle_lane import LaneCircle
from trasim_simplified.core.frame.open_lane import LaneOpen
from trasim_simplified.util.decorator.mydecorator import timer_no_log
from trasim_simplified.core.data.data_processor import Info as P_Info
from trasim_simplified.core.data.data_container import Info as C_Info


@timer_no_log
def test_circle():
    _cf_param = {"lambda": 0.8}
    take_over_index = 0
    dt = 1
    warm_up_step = 0
    sim_step = warm_up_step + int(1200 / dt)
    offset_step = int(5 / dt)
    is_circle = False

    if is_circle:
        sim = LaneCircle(1000)
        sim.car_config(40, 5, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param)
        # sim.car_config(40, 5, V_TYPE.PASSENGER, 0, False, CFM.GIPPS, _cf_param)
        sim.car_load()
    else:
        sim = LaneOpen(1000)
        sim.car_config(40, 5, V_TYPE.PASSENGER, 20, False, CFM.GIPPS, _cf_param)
        sim.car_loader(2000)

    sim.data_container.config()
    sim.data_processor.config()
    for step in sim.run(data_save=True, has_ui=True, frame_rate=-1, warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        # pass
        if warm_up_step + offset_step == step:
            take_over_index = sim.get_appropriate_car()
        if warm_up_step + offset_step < step <= warm_up_step + offset_step + 20 / dt:
            sim.take_over(take_over_index, -3)
    sim.data_container.data_to_df()

    print(f"TET_sum: {np.sum(sim.data_container.data_df[C_Info.safe_tet])}")
    print(f"TIT_sum: {np.sum(sim.data_container.data_df[C_Info.safe_tit])}")

    follower_index = (take_over_index - 1) if take_over_index != 0 else sim.car_num - 1

    sim.plot.basic_plot(follower_index)
    sim.plot.spatial_time_plot(follower_index, C_Info.v)
    sim.plot.show()

    # sim.data_processor.aggregate_as_detect_loop(0, 995, 6000)
    # sim.data_processor.print_result()


if __name__ == '__main__':
    test_circle()
