# -*- coding: utf-8 -*-
# @time : 2025/4/9 12:20
# @Author : yzbyx
# @File : run_scenario_sim_platoon.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from skopt import gp_minimize
from skopt.space import Real

from trasim_simplified.core.agent.game_agent import Game_Vehicle, Game_A_Vehicle, Game_H_Vehicle
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM, V_CLASS, RouteType, MARKING_TYPE, SECTION_TYPE
from trasim_simplified.core.frame.micro.road import Road
# from trasim_simplified.util.scenario_plot import plot_stra
from trasim_simplified.util.timer import timer_no_log
from trasim_simplified.core.data.data_container import Info as C_Info
from trasim_simplified.util.tools import save_to_pickle, load_from_pickle


def make_road():
    road_length = 600
    upstream_ratio = 1 / 3
    weaving_ratio = 1 / 3
    upstream_end = int(road_length * upstream_ratio)
    downstream_start = int(road_length * (upstream_ratio + weaving_ratio))

    lane_num = 2
    road = Road(road_length)
    road.set_start_weaving_pos(upstream_end)
    road.set_end_weaving_pos(downstream_start)

    lanes = road.add_lanes(lane_num, is_circle=False)
    for i in range(lane_num):
        if i == 0:
            lanes[i].set_marking_type(
                [
                    (MARKING_TYPE.SOLID, MARKING_TYPE.SOLID),
                    (MARKING_TYPE.SOLID, MARKING_TYPE.DASHED),
                    (MARKING_TYPE.SOLID, MARKING_TYPE.SOLID),
                ],
                [0, upstream_end, downstream_start, road_length],
            )
        elif i == lane_num - 1:
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
        else:
            raise ValueError("Invalid lane index")
    return road


@timer_no_log
def run_road(road: Road):
    dt = 0.1
    warm_up_step = 0
    sim_step = warm_up_step + int(20 / dt)

    v_length = 5

    lanes = road.lane_list
    lane_num = len(lanes)
    upstream_end = road.start_weaving_pos

    save_info = [C_Info.trackId, C_Info.frame, C_Info.time, C_Info.length, C_Info.width,
                 C_Info.xCenterGlobal, C_Info.yCenterGlobal,
                 C_Info.speed, C_Info.acc, C_Info.yaw, C_Info.delta,
                 C_Info.lane_add_num, C_Info.ttc]

    for i in range(lane_num):
        lanes[i].data_container.config(save_info=save_info, basic_info=False)

    speed = 10
    v0 = 10
    dhw = speed * 1.3 + v_length + 2

    EV_pos = upstream_end - 20
    TP_pos = EV_pos + dhw
    CP_pos = EV_pos + dhw

    veh_TP: Game_Vehicle = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        TP_pos, speed, 0,
        CFM.KK, {"v0": v0}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TP.no_lc = True
    veh_TP.rho = 0.5
    # veh_TP.game_co = 0.5

    veh_EV: Game_A_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_AV,
        EV_pos, speed, 0,
        CFM.TPACC, {"v0": 10}, {"color": COLOR.green},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.merge
    )
    # veh_EV.no_lc = True
    veh_EV.rho = 0.5
    veh_EV.can_raise_game = True

    veh_CP: Game_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        CP_pos, speed, 0,
        CFM.KK, {"v0": v0}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[1], route_type=RouteType.auxiliary
    )
    veh_CP.no_lc = True

    has_ui = True

    TR_stra_dict = {}
    TP_stra_dict = {}
    for step, stage in sim.run(data_save=True, has_ui=has_ui, frame_rate=-1,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        print(step, stage)
        if stage == 4:
            # sim.ui.plot_pred_traj()
            # sim.ui.plot_hist_traj()
            plt.pause(0.01)

    # plot_stra(TR_stra_dict, sim_step, save_file_name, type_="TR")
    # plot_stra(TP_stra_dict, sim_step, save_file_name, type_="TP")

    df = sim.data_to_df()
    df[C_Info.localLonAcc] = df[C_Info.acc] * np.cos(df[C_Info.yaw])
    df[C_Info.localLatAcc] = df[C_Info.acc] * np.sin(df[C_Info.yaw])
    df[C_Info.localLonVel] = df[C_Info.speed] * np.cos(df[C_Info.yaw])
    df[C_Info.localLatVel] = df[C_Info.speed] * np.sin(df[C_Info.yaw])

    traj_s = []
    traj_names = ["EV", "TR", "TF", "PC", "TRR", "TPP"]
    for i in [veh_EV.ID, veh_TR.ID, veh_TP.ID, veh_CP.ID, veh_TRR.ID, veh_TPP.ID]:
        traj = df[df[C_Info.trackId] == i].sort_values(C_Info.frame)
        traj_s.append(traj)

    save_to_pickle([traj_s, traj_names], rf"data\{save_file_name}_traj_s.pkl")


if __name__ == '__main__':
    sim = make_road()
    run_road(sim)
