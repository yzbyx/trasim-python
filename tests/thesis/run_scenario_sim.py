# -*- coding: utf-8 -*-
# @time : 2025/4/9 12:20
# @Author : yzbyx
# @File : run_scenario_sim.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from trasim_simplified.core.agent.game_agent import Game_Vehicle, Game_A_Vehicle
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
    dhw += 5

    x_base = upstream_end + 50

    veh_TRR = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        x_base - dhw * 2, 10, 0,
        CFM.KK, {"v0": 10}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TRR.no_lc = True
    veh_TRR.rho = 0.1

    veh_TR = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        x_base - dhw, 10, 0,
        CFM.KK, {"v0": 10}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TR.no_lc = True
    veh_TR.rho = 0.9
    # veh_TR.game_co = 0

    veh_TF: Game_Vehicle = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        x_base, 10, 0,
        CFM.KK, {"v0": 10}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TF.no_lc = True

    veh_EV: Game_A_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_AV,
        x_base - dhw, 10, 0,
        CFM.TPACC, {"v0": 10}, {"color": COLOR.green},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.merge
    )
    # veh_EV.no_lc = True
    veh_EV.rho = 0.5

    veh_PC: Game_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        x_base, 10, 0,
        CFM.KK, {"v0": 10}, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[1], route_type=RouteType.auxiliary
    )
    veh_PC.no_lc = True

    # for veh in [veh_TRR, veh_TR]:
    #     veh.single_stra = True

    has_ui = True

    lc_cross_time = []
    lc_end_time = []
    TR_stra_dict = {}
    for step, stage in sim.run(data_save=True, has_ui=has_ui, frame_rate=-1,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        # if stage == 2:
        #     veh_TR.is_gaming = True
        #     veh_TR.game_factor = 1 / 3

        if stage == 4:
            print(veh_TR.ID, veh_TR.is_gaming, veh_TR.game_factor)
            sim.ui.focus_on(veh_EV)
            sim.ui.plot_pred_traj()
            sim.ui.plot_hist_traj()
            plt.pause(0.01)

            print("-" * 10 + "basic_info" + "-" * 10)
            print("step:", step, veh_EV)
            if veh_EV.gap_res_list is not None:
                print("-" * 10 + "gap_res_list" + "-" * 10)
                for res in veh_EV.gap_res_list:
                    print(res)

            if veh_EV.opti_gap is not None:
                print("-" * 10 + "opti_gap_res" + "-" * 10)
                print(veh_EV.opti_gap)

            if veh_EV.game_res_list is not None:
                print("-" * 10 + "game_res_list" + "-" * 10)
                for res in veh_EV.game_res_list:
                    print(res)

            if veh_EV.opti_game_res is not None:
                print("-" * 10 + "opti_game_res" + "-" * 10)
                print(veh_EV.opti_game_res)
                print(veh_EV.rho_hat_s)
                if veh_EV.opti_game_res.TR_stra is not None:
                    tr_id = veh_EV.opti_game_res.TR.ID
                    if isinstance(veh_EV.opti_game_res.TR, Game_A_Vehicle):
                        rho = veh_EV.opti_game_res.TR.rho
                    else:
                        rho = np.mean(veh_EV.rho_hat_s[tr_id])
                    TR_stra_dict[step] = \
                        (tr_id, veh_EV.opti_game_res.TR_stra, rho)

            print(veh_EV.lc_conti_time)

    # 绘制激进度估计与TR策略变化曲线
    tr_id_s = []
    tr_stra_s = []
    tr_rho_s = []
    if TR_stra_dict is not None:
        steps = np.arange(sim_step)
        for step in steps:
            if step in TR_stra_dict:
                tr_id, tr_stra, rho = TR_stra_dict[step]
            else:
                tr_id = np.nan
                tr_stra = np.nan
                rho = np.nan
            tr_id_s.append(tr_id)
            tr_stra_s.append(tr_stra)
            tr_rho_s.append(rho)
        tr_id_s = np.array(tr_id_s)
        tr_stra_s = np.array(tr_stra_s)
        tr_rho_s = np.array(tr_rho_s)

        # 设置坐标轴范围
        mm = 1 / 25.4  # mm转inch
        _width = 70 * mm * 2  # 图片宽度英寸
        _ratio = 3 / 14  # 图片长宽比
        figsize = (_width, _width * _ratio)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig: plt.Figure = fig
        ax: plt.Axes = ax
        colors = seaborn.color_palette('Set1', n_colors=3)
        # 根据tr_id的数量对stra和rho进行绘制
        tr_ids_unique = np.unique(tr_id_s)
        tr_ids_unique = tr_ids_unique[~np.isnan(tr_ids_unique) & (tr_ids_unique > 0)]
        for i, tr_id in enumerate(tr_ids_unique):
            indexes = np.where(tr_id_s == tr_id)[0]
            tr_stra_temp = np.zeros(len(steps)) * np.nan
            tr_stra_temp[indexes] = tr_stra_s[indexes]
            tr_rho_temp = np.zeros(len(steps)) * np.nan
            tr_rho_temp[indexes] = tr_rho_s[indexes]
            ax.plot(steps * 0.1, tr_stra_temp, label=r"$s_{TR}$" + str(int(tr_id)), color=colors[i])
            ax.plot(steps * 0.1, tr_rho_temp, label=r"$\rho_{TR}$" + str(int(tr_id)),
                    color=colors[i], linestyle='--')

        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("数值")
        ax.legend(fontsize=9)

        fig.savefig(
            fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\merging_HV_HV_TR_stra.png",
            dpi=300, bbox_inches='tight'
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
