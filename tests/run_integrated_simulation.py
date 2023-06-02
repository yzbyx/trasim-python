# -*- coding: utf-8 -*-
# @Time : 2023/5/31 23:55
# @Author : yzbyx
# @File : run_integrated_simulation.py
# Software: PyCharm
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.open_lane import THW_DISTRI
from trasim_simplified.core.frame.road import Road
from trasim_simplified.util.decorator.mydecorator import timer_no_log
from trasim_simplified.core.data.data_container import Info as C_Info


@timer_no_log
def run_integrated_simulation():
    _cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True}
    _car_param = {}
    take_over_index = -1
    follower_index = -1
    dt = 1
    warm_up_step = 0
    offset_step = int(300 / dt)
    sim_step = warm_up_step + int(1800 / dt)

    road_length = 10000 + 10000
    lane_num = 1

    sim = Road(road_length)
    lanes = sim.add_lanes(lane_num, is_circle=False)
    lanes[0].set_speed_limit(30)
    lanes[0].data_container.config()
    lanes[0].car_insert(6, V_TYPE.PASSENGER, )

    for step, state in sim.run(data_save=True, has_ui=False, frame_rate=-1,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        if warm_up_step + offset_step == step and state == 0:
            take_over_index = sim.get_appropriate_car(lane_index=0)
            print(take_over_index)
        # if warm_up_step + offset_step <= step < warm_up_step + offset_step + int(60 / dt):
        #     sim.take_over(take_over_index, -3, lc_result={"lc": 0})

    df = sim.data_to_df()
    lane_ids = sim.find_on_lanes(take_over_index)
    Plot.basic_plot(take_over_index, lane_id=lane_ids[0], data_df=df)
    Plot.spatial_time_plot(take_over_index, lane_id=0,
                           color_info_name=C_Info.v, data_df=df, single_plot=False)
    Plot.spatial_time_plot(take_over_index, lane_id=1,
                           color_info_name=C_Info.v, data_df=df, single_plot=False)
    # Plot.spatial_time_plot(take_over_index, lane_id=2,
    #                        color_info_name=C_Info.v, data_df=df, single_plot=False)
    Plot.show()