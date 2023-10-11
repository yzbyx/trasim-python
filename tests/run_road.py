# -*- coding: utf-8 -*-
# @Time : 2023/5/12 21:54
# @Author : yzbyx
# @File : run_road.py
# Software: PyCharm
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.micro.open_lane import THW_DISTRI
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.util.decorator.timer import timer_no_log
from trasim_simplified.core.data.data_container import Info as C_Info


@timer_no_log
def run_road():
    _cf_param = {"lambda": 0.8, "original_acc": True, "v_safe_dispersed": True}
    _car_param = {}
    take_over_index = -1
    follower_index = -1
    dt = 1
    warm_up_step = 0
    sim_step = warm_up_step + int(3600 / dt)
    offset_step = int(300 / dt)

    is_circle = False
    road_length = 15000
    lane_num = 1

    if is_circle:
        sim = Road(road_length)
        lanes = sim.add_lanes(lane_num, is_circle=True)
        for i in range(lane_num):
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.yellow},
                                lc_name=LCM.KK, lc_param={})
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.blue},
                                lc_name=LCM.KK, lc_param={})
            lanes[i].car_load()
            lanes[i].data_container.config()
    else:
        sim = Road(road_length)
        lanes = sim.add_lanes(lane_num, is_circle=False)
        for i in range(lane_num):
            # lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.yellow},
            #                     lc_name=LCM.KK, lc_param={})
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, 20, False, CFM.TPACC, _cf_param, {"color": COLOR.blue},
                                lc_name=LCM.ACC, lc_param={})
            if i == 0:
                lanes[i].car_loader(2000, THW_DISTRI.Uniform)
            else:
                lanes[i].car_loader(400, THW_DISTRI.Uniform)
            lanes[i].data_container.config()

    for step, state in sim.run(data_save=True, has_ui=True, frame_rate=-1,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        pass
        # if step == 0 and state == 0:
        #     leader_id = sim.lane_list[0].car_insert(7.5, V_TYPE.PASSENGER, road_length / 2, 30,
        #                                             0, CFM.KK, _cf_param, {"color": COLOR.blue},
        #                                             lc_name=LCM.ACC, lc_param={})
        #     dist = 20
        #     follower_id = sim.lane_list[0].car_insert(7.5, V_TYPE.PASSENGER, road_length / 2 - dist, 30,
        #                                               0, CFM.KK, _cf_param, {"color": COLOR.blue},
        #                                               lc_name=LCM.ACC, lc_param={})
        #     insert_id = sim.lane_list[0].car_insert(7.5, V_TYPE.PASSENGER, road_length / 2 - dist / 2, 30,
        #                                             0, CFM.ACC, _cf_param, {"color": COLOR.blue},
        #                                             lc_name=LCM.ACC, lc_param={})
        # sim.take_over(take_over_index, 0, lc_result={"lc": 0})
        # if step == 2:
        #     back_id = sim.lane_list[0].get_relative_id(take_over_index, -1)
        # if warm_up_step + offset_step == step and state == 0:
        #     take_over_index = sim.get_appropriate_car(lane_add_num=0)
        #     print(take_over_index)
        # if warm_up_step + offset_step <= step < warm_up_step + offset_step + int(60 / dt):
        #     sim.take_over(take_over_index, -3, lc_result={"lc": 0})

    df = sim.data_to_df()
    df.to_csv("D:\\test_df.csv")
    # lane_ids = sim.find_on_lanes(follower_id)
    # Plot.basic_plot(follower_id, lane_id=lane_ids[0], data_df=df)
    # Plot.basic_plot(insert_id, lane_id=lane_ids[0], data_df=df)
    # Plot.spatial_time_plot(insert_id, lane_add_num=0,
    #                        color_info_name=C_Info.v, data_df=df, single_plot=False)
    # Plot.spatial_time_plot(take_over_index, lane_add_num=1,
    #                        color_info_name=C_Info.v, data_df=df, single_plot=False)
    # Plot.spatial_time_plot(take_over_index, lane_id=2,
    #                        color_info_name=C_Info.v, data_df=df, single_plot=False)
    # Plot.show()

    # sim.data_processor.aggregate_as_detect_loop(0, 995, 6000)
    # sim.data_processor.print_result()


if __name__ == '__main__':
    run_road()
