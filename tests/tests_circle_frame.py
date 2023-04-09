# -*- coding = uft-8 -*-
# @Time : 2023-04-09 20:20
# @Author : yzbyx
# @File : tests_circle_frame.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.core.constant import CFM
from trasim_simplified.core.frame.circle_frame import FrameCircle
from trasim_simplified.util.decorator.mydecorator import timer_no_log
from trasim_simplified.core.data.data_processor import Info as P_Info


@timer_no_log
def test_circle():
    _cf_param = {"lambda": 1 / np.e}
    take_over_index = 0
    warm_up_step = 6000
    sim_step = 12000
    dt = 1.
    sim = FrameCircle(1000, 20, 5, 10, False, CFM.LINEAR, _cf_param)
    sim.data_container.config()
    sim.data_processor.config()
    sim.data_processor.ttc_star = 3
    for step in sim.run(data_save=True, has_ui=False, frame_rate=120, warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        if warm_up_step + 100 == step:
            take_over_index = sim.get_appropriate_car()
        if warm_up_step + 100 < step <= warm_up_step + 100 + 2 / dt:
            sim.take_over(take_over_index, -3)
    result = sim.data_processor.cal_safety()
    print(f"TET_sum: {np.sum(result[P_Info.safe_tet])}")
    print(f"TIT_sum: {np.sum(result[P_Info.safe_tit])}")

    sim.plot.basic_plot(((take_over_index - 1) if take_over_index != 0 else sim.car_num - 1))
    # sim.data_processor.aggregate_as_detect_loop(0, 995, 6000)
    # sim.data_processor.print_result()
