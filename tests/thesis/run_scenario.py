# -*- coding: utf-8 -*-
# @Time : 2025/4/9 12:20
# @Author : yzbyx
# @File : run_scenario.py
# Software: PyCharm
import matplotlib.pyplot as plt

from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM, SECTION_TYPE, MARKING_TYPE, V_CLASS, RouteType
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.micro.open_lane import THW_DISTRI, LaneOpen
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

    road_length = 600
    upstream_ratio = 1 / 3
    weaving_ratio = 1 / 3
    upstream_end = int(road_length * upstream_ratio)
    downstream_start = int(road_length * (upstream_ratio + weaving_ratio))
    lane_num = 2
    v_length = 5

    sim = Road(road_length)
    lanes: list[LaneOpen] = sim.add_lanes(lane_num, is_circle=False)
    for i in range(lane_num):
        if i != lane_num - 1:
            lanes[i].set_speed_limit(30)
        else:
            lanes[i].set_speed_limit(22.2)

        if i == lane_num - 2:
            lanes[i].set_marking_type(
                [
                    (MARKING_TYPE.DASHED, MARKING_TYPE.SOLID),
                    (MARKING_TYPE.DASHED, MARKING_TYPE.DASHED),
                    (MARKING_TYPE.DASHED, MARKING_TYPE.SOLID),
                ],
                [0, upstream_end, downstream_start, road_length],
            )
        if i == lane_num - 1:
            lanes[i].set_section_type(
                [
                    SECTION_TYPE.ON_RAMP, SECTION_TYPE.AUXILIARY, SECTION_TYPE.OFF_RAMP,
                ],
                [0, upstream_end, downstream_start, road_length],
            )
            lanes[i].set_marking_type(
                [
                    (MARKING_TYPE.SOLID, MARKING_TYPE.SOLID),
                    (MARKING_TYPE.DASHED, MARKING_TYPE.SOLID),
                    (MARKING_TYPE.SOLID, MARKING_TYPE.SOLID),
                ],
                [0, upstream_end, downstream_start, road_length],
            )
        sim.set_start_weaving_pos(upstream_end)
        sim.set_end_weaving_pos(downstream_start)

        lanes[i].data_container.config()

    veh_TR = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        upstream_end - 50, 10, 0,
        CFM.IDM, _cf_param, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}
    )
    veh_TR.no_lc = True
    veh_TR.rho = 1

    veh_TF = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        upstream_end + 120, 10, 0,
        CFM.IDM, _cf_param, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}
    )
    veh_TF.no_lc = True

    veh_EV = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_AV,
        upstream_end + 40, 10, 0,
        CFM.IDM, _cf_param, {"color": COLOR.green},
        lc_name=LCM.MOBIL, lc_param={}
    )

    veh_PC = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        upstream_end + 90, 10, 0,
        CFM.IDM, _cf_param, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}
    )
    veh_PC.no_lc = True

    has_ui = True
    for step, stage in sim.run(data_save=True, has_ui=has_ui, frame_rate=-1,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        if stage == 0:
            sim.ui.focus_on(veh_EV)
            sim.ui.plot_pred_traj()
            sim.ui.plot_hist_traj()
            plt.pause(0.01)
        if warm_up_step + offset_step == step and stage == 0:
            take_over_index = sim.get_appropriate_car(lane_add_num=0)
            print(take_over_index)
        if warm_up_step + offset_step <= step < warm_up_step + offset_step + int(60 / dt):
            sim.take_over(take_over_index, -3, lc_result={"lc": 0})
        # if warm_up_step + offset_step == step and stage == 1:
        #     take_over_index = sim.get_appropriate_car(lane_index=0)
        #     print(take_over_index)
        #     for lane in sim.lane_list:
        #         lane.set_speed_limit(20, 5000, 10000)
        pass

    df = sim.data_to_df()
    # result = sim.data_processor.aggregate_as_detect_loop(
    #     df, lane_id=0, lane_length=road_length, pos=5000, width=50,
    #     dt=dt, d_step=int(300 / dt)
    # )
    # sim.data_processor.print(result)

    Plot.spatial_time_plot(df, take_over_index, lane_add_num=0,
                           color_info_name=C_Info.v, single_plot=False)
    Plot.show()


if __name__ == '__main__':
    run_road()
