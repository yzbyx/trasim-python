# -*- coding: utf-8 -*-
# @Time : 2023/5/15 17:34
# @Author : yzbyx
# @File : run_on_ramp_test.py
# Software: PyCharm
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM, SECTION_TYPE
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.micro.open_lane import THW_DISTRI
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.util.timer import timer_no_log
from trasim_simplified.core.data.data_container import Info as C_Info


@timer_no_log
def run_road():
    _cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True, "g_tau": 3, "kdv": 0.3}
    _car_param = {}
    take_over_index = -1
    follower_index = -1
    dt = 1
    warm_up_step = 0
    sim_step = warm_up_step + int(3600 / dt)
    offset_step = int(1800 / dt)

    is_circle = False
    road_length = 15000
    lane_num = 2
    v_length = 7.5

    if is_circle:
        sim = Road(road_length)
        lanes = sim.add_lanes(lane_num, is_circle=True)
        for i in range(lane_num):
            if i != lane_num - 1:
                lanes[i].car_config(200, v_length, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.yellow},
                                    lc_name=LCM.KK, lc_param={})
                lanes[i].car_config(200, v_length, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.blue},
                                    lc_name=LCM.KK, lc_param={})
            else:
                lanes[i].car_config(100, v_length, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.yellow},
                                    lc_name=LCM.KK, lc_param={})
                lanes[i].car_config(100, v_length, V_TYPE.PASSENGER, 20, False, CFM.ACC, _cf_param, {"color": COLOR.blue},
                                    lc_name=LCM.ACC, lc_param={})
            lanes[i].car_load()
            lanes[i].data_container.config()
            if i == lane_num - 2:
                lanes[i].add_section_type(SECTION_TYPE.BASE)
                lanes[i].add_section_type(SECTION_TYPE.NO_RIGHT)
            if i == lane_num - 1:
                lanes[i].add_section_type(SECTION_TYPE.ON_RAMP, 5000, -1)
                lanes[i].add_section_type(SECTION_TYPE.NO_LEFT, 0, 5000)
                lanes[i].add_section_type(SECTION_TYPE.BASE, 0, 5000)
                lanes[i].set_block(5300)
    else:
        sim = Road(road_length)
        lanes = sim.add_lanes(lane_num, is_circle=False)
        for i in range(lane_num):
            if i != lane_num - 1:
                lanes[i].set_speed_limit(30)
            else:
                lanes[i].set_speed_limit(22.2)

            if i == lane_num - 2:
                lanes[i].add_section_type(SECTION_TYPE.BASE)
                lanes[i].add_section_type(SECTION_TYPE.NO_RIGHT)
            if i == lane_num - 1:

                lanes[i].add_section_type(SECTION_TYPE.ON_RAMP, 10000, -1)
                lanes[i].add_section_type(SECTION_TYPE.NO_LEFT, 0, 10000)
                lanes[i].add_section_type(SECTION_TYPE.BASE, 0, 10000)
                lanes[i].set_block(10300)
            else:
                # lanes[i].car_load()
                pass

            lanes[i].car_config(80, v_length, V_TYPE.PASSENGER, lanes[i].get_speed_limit(0), False,
                                CFM.TPACC, _cf_param, {"color": COLOR.yellow}, lc_name=LCM.ACC, lc_param={})
            lanes[i].car_config(20, v_length, V_TYPE.PASSENGER, lanes[i].get_speed_limit(0), False,
                                CFM.KK, _cf_param, {"color": COLOR.blue}, lc_name=LCM.KK, lc_param={})

            lanes[i].data_container.config()

            if i != lane_num - 1:
                lanes[i].car_loader(2000, THW_DISTRI.Uniform)
            else:
                lanes[i].car_loader(100, THW_DISTRI.Uniform, 400)

    for step, stage in sim.run(data_save=True, has_ui=False, frame_rate=10,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        if warm_up_step + offset_step == step and stage == 0:
            take_over_index = sim.get_appropriate_car(lane_add_num=0)
            print(take_over_index)
        # if warm_up_step + offset_step <= step < warm_up_step + offset_step + int(60 / dt):
        #     sim.take_over(take_over_index, -3, lc_result={"lc": 0})
        # if warm_up_step + offset_step == step and stage == 1:
        #     take_over_index = sim.get_appropriate_car(lane_index=0)
        #     print(take_over_index)
        #     for lane in sim.lane_list:
        #         lane.set_speed_limit(20, 5000, 10000)
        pass

    df = sim.data_to_df()
    # result = sim.data_processor.aggregate_as_detect_loop(df, lane_id=0, lane_length=road_length, pos=5000, width=50,
    #                                                      dt=dt, d_step=int(300 / dt))
    # sim.data_processor.print(result)

    # lane_ids = sim.find_on_lanes(take_over_index)
    # Plot.basic_plot(take_over_index, lane_id=lane_ids[0], data_df=df)
    Plot.spatial_time_plot(take_over_index, lane_add_num=0,
                           color_info_name=C_Info.safe_picud, data_df=df, single_plot=False)
    Plot.spatial_time_plot(take_over_index, lane_add_num=0,
                           color_info_name=C_Info.v, data_df=df, single_plot=False)
    Plot.spatial_time_plot(take_over_index, lane_add_num=1,
                           color_info_name=C_Info.safe_tet, data_df=df, single_plot=False)
    # Plot.spatial_time_plot(take_over_index, lane_id=2,
    #                        color_info_name=C_Info.v, data_df=df, single_plot=False)
    Plot.show()


if __name__ == '__main__':
    run_road()
