# -*- coding = uft-8 -*-
# @Time : 2023-04-09 20:20
# @Author : yzbyx
# @File : tests_lane.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.core.constant import CFM, V_TYPE, COLOR
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.circle_lane import LaneCircle
from trasim_simplified.core.frame.open_lane import LaneOpen
from trasim_simplified.util.decorator.mydecorator import timer_no_log
from trasim_simplified.core.data.data_container import Info as C_Info


@timer_no_log
def test_circle():
    _cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True}
    _car_param = {}
    take_over_index = -1
    follower_index = -1
    dt = 1
    warm_up_step = 0
    sim_step = warm_up_step + int(1200 / dt)
    offset_step = int(600 / dt)

    is_circle = True

    if is_circle:
        sim = LaneCircle(1000)
        sim.car_config(132, 7.5, V_TYPE.PASSENGER, 0, False, CFM.TPACC, _cf_param, {"color": COLOR.yellow})
        # sim.car_config(50, 7.5, V_TYPE.PASSENGER, 20, False, CFM.NON_LINEAR_GHR, _cf_param, {"color": COLOR.blue})
        sim.car_load(0)
        # sim.set_block(800)
    else:
        sim = LaneOpen(10000)
        sim.car_config(40, 7.5, V_TYPE.PASSENGER, -1, False, CFM.KK, _cf_param, _car_param)
        sim.car_loader(2000)

    sim.data_container.config()
    for step in sim.run(data_save=True, has_ui=False, frame_rate=10,
                        warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        # 车辆减速扰动
        # if warm_up_step + offset_step == step:
        #     take_over_index = sim.get_appropriate_car()
        #     follower_index = sim.get_relative_id(take_over_index, -1)
        #     print(take_over_index, follower_index)
        # if warm_up_step + offset_step < step <= warm_up_step + offset_step + 100 / dt:
        #     sim.take_over(take_over_index, -3)

        # 居中插入车辆
        # if warm_up_step + offset_step == step:
        #     take_over_index = sim.get_appropriate_car()
        #     take_over_speed = sim.get_car_info(take_over_index, C_Info.v)
        #     follower_index = sim.car_insert_middle(
        #         7.5, V_TYPE.PASSENGER, take_over_speed, 0, CFM.IDM,
        #         _cf_param, {"color": COLOR.yellow}, take_over_index
        #     )
        #
        #     print(take_over_index, follower_index)

        pass

    df = sim.data_container.data_to_df()

    print(f"TET_sum: {np.sum(sim.data_container.data_df[C_Info.safe_tet])}")
    print(f"TIT_sum: {np.sum(sim.data_container.data_df[C_Info.safe_tit])}")

    Plot.basic_plot(follower_index, lane_id=0, data_df=df)
    Plot.spatial_time_plot(follower_index, lane_id=0, color_info_name=C_Info.v, data_df=df)
    Plot.show()

    # sim.data_processor.aggregate_as_detect_loop(0, 995, 6000)
    # sim.data_processor.print_result()


if __name__ == '__main__':
    test_circle()
