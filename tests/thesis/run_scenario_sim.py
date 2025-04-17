# -*- coding: utf-8 -*-
# @time : 2025/4/9 12:20
# @Author : yzbyx
# @File : run_scenario_sim.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

from trasim_simplified.core.agent.game_agent import Game_Vehicle
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM, V_CLASS, RouteType, MARKING_TYPE, SECTION_TYPE
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.core.kinematics.cfm.CFModel_IDM import cf_IDM_equilibrium_jit
from trasim_simplified.util.scenario.util import make_road_from_osm
from trasim_simplified.util.scenario_plot import plot_scenario
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
        if i != lane_num - 1:
            lanes[i].set_speed_limit(30)
        else:
            lanes[i].set_speed_limit(22.2)

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

        road.set_start_weaving_pos(upstream_end)
        road.set_end_weaving_pos(downstream_start)
    return road


@timer_no_log
def run_road(road: Road):
    dt = 0.1
    warm_up_step = 0
    sim_step = warm_up_step + int(15 / dt)

    v_length = 5

    lanes = road.lane_list
    lane_num = len(lanes)
    upstream_end = road.start_weaving_pos

    save_info = [C_Info.trackId, C_Info.frame, C_Info.time, C_Info.length, C_Info.width,
                 C_Info.xCenterGlobal, C_Info.yCenterGlobal,
                 C_Info.speed, C_Info.acc, C_Info.yaw, C_Info.delta,
                 C_Info.lane_add_num]

    for i in range(lane_num):
        lanes[i].data_container.config(save_info=save_info, basic_info=False)

    dhw = cf_IDM_equilibrium_jit(s0=2, s1=0, v0=15, T=1.6, delta=4, v=10) + v_length

    x_base = upstream_end + 50

    veh_TRR = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        x_base - dhw * 2, 10, 0,
        CFM.IDM, {"v0": 15}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TRR.no_lc = True
    veh_TRR.rho = 0.9

    veh_TR = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        x_base - dhw, 10, 0,
        CFM.IDM, {"v0": 15}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TR.no_lc = True
    veh_TR.rho = 0.9

    veh_TF: Game_Vehicle = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        x_base, 10, 0,
        CFM.IDM, {"v0": 10}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TF.no_lc = True

    veh_EV: Game_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_AV,
        x_base - dhw - 5, 10, 0,
        CFM.IDM, {"v0": 15}, {"color": COLOR.green},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.merge
    )
    # veh_EV.no_lc = True

    veh_PC: Game_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        x_base - 5, 10, 0,
        CFM.IDM, {"v0": 15}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[1], route_type=RouteType.auxiliary
    )
    veh_PC.no_lc = True

    for veh in [veh_TRR, veh_TR]:
        veh.single_stra = True

    has_ui = True

    lc_cross_time = []
    lc_end_time = []
    for step, stage in sim.run(data_save=True, has_ui=has_ui, frame_rate=-1,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        if stage == 0:
            sim.ui.focus_on(veh_EV)
            sim.ui.plot_pred_traj()
            sim.ui.plot_hist_traj()
            plt.pause(0.01)
            if veh_EV.opti_game_res is not None:
                TF = veh_EV.opti_game_res.TF
                TR = veh_EV.opti_game_res.TR
                print(
                    "game_TF", TF.ID, "game_TR", TR.ID,
                    "game_time_wanted", TR.game_time_wanted,
                    "rho_hat", veh_EV.rho_hat_s[TR.ID],
                )

    df = sim.data_to_df()

    df[C_Info.localLonAcc] = df[C_Info.acc] * np.cos(df[C_Info.yaw])
    df[C_Info.localLatAcc] = df[C_Info.acc] * np.sin(df[C_Info.yaw])
    df[C_Info.localLonVel] = df[C_Info.speed] * np.cos(df[C_Info.yaw])
    df[C_Info.localLatVel] = df[C_Info.speed] * np.sin(df[C_Info.yaw])

    traj_s = []
    for i in [veh_EV.ID, veh_TR.ID, veh_TF.ID, veh_PC.ID, veh_TRR.ID]:
        traj = df[df[C_Info.trackId] == i].sort_values(C_Info.frame)
        traj_s.append(traj)

    save_to_pickle(traj_s, r"data\merging_HV_HV_traj_s.pkl")


if __name__ == '__main__':
    sim = make_road()
    run_road(sim)
    traj_s = load_from_pickle(r"data\merging_HV_HV_traj_s.pkl")
    plot_scenario(traj_s, traj_names=["EV", "TR", "TF", "PC", "TRR"], road=sim, fig_name="merging_HV_HV")
