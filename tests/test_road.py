# -*- coding: utf-8 -*-
# @Time : 2023/5/12 21:54
# @Author : yzbyx
# @File : test_road.py
# Software: PyCharm
import numpy as np

from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.open_lane import THW_DISTRI
from trasim_simplified.core.frame.road import Road
from trasim_simplified.util.decorator.mydecorator import timer_no_log
from trasim_simplified.core.data.data_container import Info as C_Info

@timer_no_log
def test_road():
    _cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True}
    _car_param = {}
    take_over_index = -1
    follower_index = -1
    dt = 1
    warm_up_step = 0
    sim_step = warm_up_step + int(1200 / dt)
    offset_step = int(50 / dt)

    is_circle = True
    road_length = 1000
    lane_num = 2

    if is_circle:
        sim = Road(road_length)
        lanes = sim.add_lanes(lane_num, is_circle=True)
        for i in range(lane_num):
            lanes[i].car_config(20, 7.5, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.yellow},
                                lc_name=LCM.KK, lc_param={})
            lanes[i].car_config(20, 7.5, V_TYPE.PASSENGER, 20, False, CFM.TPACC, _cf_param, {"color": COLOR.blue},
                                lc_name=LCM.KK, lc_param={})
            lanes[i].car_load()
            lanes[i].data_container.config()
            lanes[i].data_processor.config()
    else:
        sim = Road(road_length)
        lanes = sim.add_lanes(lane_num, is_circle=False)
        for i in range(lane_num):
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, -1, False, CFM.KK, _cf_param, {"color": COLOR.yellow},
                                lc_name=LCM.KK, lc_param={})
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, -1, False, CFM.TPACC, _cf_param, {"color": COLOR.blue},
                                lc_name=LCM.ACC, lc_param={})
            lanes[i].car_loader(2000, THW_DISTRI.Exponential)
            lanes[i].data_container.config()
            lanes[i].data_processor.config()

    for step in sim.run(data_save=True, has_ui=True, frame_rate=-1,
                        warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        if warm_up_step + offset_step == step:
            take_over_index = sim.get_appropriate_car(lane_index=0)
        if warm_up_step + offset_step <= step < warm_up_step + offset_step + int(20 / dt):
            sim.take_over(take_over_index, -3, lc_result={"lc": 0})

    df = sim.data_to_df()
    lane_ids = sim.find_on_lanes(take_over_index)
    Plot.basic_plot(take_over_index, lane_id=lane_ids[0], data_df=df)
    Plot.spatial_time_plot(take_over_index, lane_id=lane_ids[0], color_info_name=C_Info.v, data_df=df, single_plot=False)
    Plot.show()
    # sim.data_processor.aggregate_as_detect_loop(0, 995, 6000)
    # sim.data_processor.print_result()


if __name__ == '__main__':
    test_road()
