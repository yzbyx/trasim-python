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

from trasim_simplified.core.agent.game_agent import Game_Vehicle, Game_A_Vehicle, Game_H_Vehicle, Game_O_Vehicle
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM, V_CLASS, RouteType, MARKING_TYPE, SECTION_TYPE
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.core.kinematics.cfm.CFModel_IDM import cf_IDM_equilibrium_jit
from trasim_simplified.util.scenario.scenario_util import make_road_from_osm
from trasim_simplified.util.scenario_plot import plot_scenario, plot_stra, plot_scenario_twin
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
                 C_Info.lane_add_num, C_Info.ttc]

    for i in range(lane_num):
        lanes[i].data_container.config(save_info=save_info, basic_info=False)

    speed = 10
    v0 = 10
    g_tau = 3
    safe_tau = 1
    s0 = 2
    scale = 0.3
    av_cf_params = {"v0": v0, "g_tau": g_tau, "safe_tau": safe_tau * scale, "s0": s0 * scale}
    hv_cf_params = {"v0": v0, "g_time": g_tau, "safe_time": safe_tau * scale, "s0": s0 * scale}
    dhw = speed * 2 + v_length * 2  # 30m

    EV_pos = upstream_end + 50
    # TR_pos = EV_pos - dhw * 2 / 3
    # TP_pos = EV_pos + dhw * 2 / 3
    # CR_pos = EV_pos - dhw * 4 / 3
    # CP_pos = EV_pos + dhw * 4 / 3
    TR_pos = EV_pos - dhw * 4 / 3
    TP_pos = EV_pos + dhw * 2 / 3
    CR_pos = EV_pos - dhw * 2 / 3
    CP_pos = EV_pos + dhw

    TRR_pos = TR_pos - dhw
    TPP_pos = TP_pos + dhw
    CPP_pos = CP_pos + dhw
    CRR_pos = CR_pos - dhw

    veh_TRR = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_OV,
        TRR_pos, speed, 0,
        CFM.KK, hv_cf_params, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TRR.no_lc = True

    veh_TPP: Game_Vehicle = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_OV,
        TPP_pos, speed, 0,
        CFM.KK, hv_cf_params, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TPP.no_lc = True

    veh_CPP: Game_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        CPP_pos, speed, 0,
        CFM.KK, hv_cf_params, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[1], route_type=RouteType.auxiliary
    )
    veh_CPP.no_lc = True

    veh_CRR: Game_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        CRR_pos, speed, 0,
        CFM.KK, hv_cf_params, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[1], route_type=RouteType.auxiliary
    )
    veh_CRR.no_lc = True

    veh_TP: Game_Vehicle = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        TP_pos, speed, 0,
        CFM.KK, hv_cf_params, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={}, destination_lanes=[0], route_type=RouteType.mainline
    )
    veh_TP.no_lc = True
    veh_TP.rho = 0.5
    # veh_TP.game_co = 0.5

    # 有交互车辆
    veh_TR = lanes[0].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_AV,
        TR_pos, speed, 0,
        CFM.TPACC, av_cf_params, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={},
        destination_lanes=[0], route_type=RouteType.mainline,
        # destination_lanes=[1], route_type=RouteType.diverge,
    )
    # veh_TR.no_lc = True
    veh_TR.rho = 0.5
    veh_TR.game_co = 1

    veh_EV: Game_A_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_AV,
        EV_pos, speed, 0,
        CFM.TPACC, av_cf_params, {"color": COLOR.green},
        lc_name=LCM.MOBIL, lc_param={},
        destination_lanes=[0], route_type=RouteType.merge
    )
    # veh_EV.no_lc = True
    veh_EV.rho = 0.5
    veh_EV.can_raise_game = True

    veh_CR: Game_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_AV,
        CR_pos, speed, 0,
        CFM.TPACC, hv_cf_params, {"color": COLOR.green},
        lc_name=LCM.MOBIL, lc_param={},
        destination_lanes=[0], route_type=RouteType.merge
    )
    veh_CR.no_lc = False
    veh_CR.game_co = 1

    veh_CP: Game_Vehicle = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        CP_pos, speed, 0,
        CFM.KK, hv_cf_params, {"color": COLOR.red},
        lc_name=LCM.MOBIL, lc_param={},
        destination_lanes=[1], route_type=RouteType.auxiliary,
        # destination_lanes=[0], route_type=RouteType.merge,
    )
    veh_CP.no_lc = True

    has_ui = True

    TR_stra_dict = {}
    TP_stra_dict = {}
    for step, stage in sim.run(data_save=True, has_ui=has_ui, frame_rate=-1,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        if stage == 4:
            # sim.ui.plot_pred_traj()
            # sim.ui.plot_hist_traj()
            plt.pause(0.01)

            for veh in sim.get_total_car():
                if isinstance(veh, Game_O_Vehicle):
                    continue
                print("-" * 30 + f"veh: {veh.name}" + "-" * 30)
                print("-" * 10 + "basic_info" + "-" * 10)
                print("step:", step, veh)
                if veh.gap_res_list is not None:
                    print("-" * 10 + "gap_res_list" + "-" * 10)
                    for res in veh.gap_res_list:
                        print(res)

                if veh.opti_gap is not None:
                    print("-" * 10 + "opti_gap_res" + "-" * 10)
                    print(veh.opti_gap)

                if veh.game_res_list is not None:
                    print("-" * 10 + "game_res_list" + "-" * 10)
                    for res in veh.game_res_list:
                        print(res)

                if veh.opti_game_res is not None:
                    print("-" * 10 + "opti_game_res" + "-" * 10)
                    print(veh.opti_game_res)
                    print(veh.rho_hat_s)
                    if veh.opti_game_res.TR_stra is not None:
                        tr_id = veh.opti_game_res.game_surr.TR.ID
                        if isinstance(veh.opti_game_res.game_surr.TR, Game_A_Vehicle):
                            rho = veh.opti_game_res.game_surr.TR.rho
                        else:
                            rho = np.mean(veh.rho_hat_s[tr_id])
                        TR_stra_dict[step] = \
                            (tr_id, veh.opti_game_res.TR_stra, rho)
                    if veh.opti_game_res.TF_stra is not None:
                        tp_id = veh.opti_game_res.game_surr.TP.ID
                        if isinstance(veh.opti_game_res.game_surr.TP, Game_A_Vehicle):
                            rho = veh.opti_game_res.game_surr.TP.rho
                        else:
                            rho = np.mean(veh.rho_hat_s[tp_id])
                        TP_stra_dict[step] = \
                            (tp_id, veh.opti_game_res.TF_stra, rho)

                # print(veh.last_cal_step)

    is_TR_HV = isinstance(veh_TR, Game_H_Vehicle)
    is_TP_HV = isinstance(veh_TP, Game_H_Vehicle)
    save_file_name = f"platoon_AV"
    # plot_stra(TR_stra_dict, sim_step, save_file_name, type_="TR")
    # plot_stra(TP_stra_dict, sim_step, save_file_name, type_="TP")

    df = sim.data_to_df()
    df[C_Info.localLonAcc] = df[C_Info.acc] * np.cos(df[C_Info.yaw])
    df[C_Info.localLatAcc] = df[C_Info.acc] * np.sin(df[C_Info.yaw])
    df[C_Info.localLonVel] = df[C_Info.speed] * np.cos(df[C_Info.yaw])
    df[C_Info.localLatVel] = df[C_Info.speed] * np.sin(df[C_Info.yaw])

    traj_s = []
    traj_names = ["EV", "TR", "TRR", "TP", "TPP", "CR", "CRR", "CP"]
    for i in [veh_EV.ID, veh_TR.ID, veh_TRR.ID, veh_TP.ID, veh_TPP.ID, veh_CR.ID, veh_CRR.ID, veh_CP.ID]:
        traj = df[df[C_Info.trackId] == i].sort_values(C_Info.frame)
        traj_s.append(traj)

    save_to_pickle([traj_s, traj_names], rf"data\{save_file_name}_traj_s.pkl")


def plot_data(sim, save_file_name):
    traj_s, traj_names = load_from_pickle(rf"data\{save_file_name}_traj_s.pkl")
    plot_scenario_twin(
        traj_s, traj_names=traj_names,
        road=sim, fig_name=save_file_name, ori_plot=False,
        highlight=["EV", "CR"]
    )

    ttc_s = []
    eff_s = []
    jerk_s = []
    acc_s = []

    for name, traj in zip(traj_names, traj_s):
        ttc = traj[C_Info.ttc].to_numpy()
        min_ttc = np.nanmin(ttc)
        tit = sum(abs(ttc[ttc < 1.3])) * 0.1
        ttc_s.append(min_ttc)
        print(f"{name}安全：", min_ttc, tit)

        v = traj[C_Info.speed].to_numpy()
        eff = v[0] - np.nanmean(v)
        eff_s.append(eff)
        print(f"{name}效率成本：", eff)

        jerk = np.diff(traj[C_Info.acc].to_numpy())
        jerk_max = np.nanmax(np.abs(jerk))
        jerk_s.append(jerk_max)
        print(f"{name}舒适成本jerk：", jerk_max)

        acc = traj[C_Info.acc].to_numpy()
        acc_max = np.nanmax(np.abs(acc))
        acc_s.append(acc_max)
        print(f"{name}舒适成本acc：", acc_max)

    print("最小ttc：", min(ttc_s))
    print("平均效率成本：", np.mean(eff_s))
    print("平均舒适成本jerk：", np.mean(jerk_s))
    print("平均舒适成本acc：", np.mean(acc_s))


def opti_ade(self, cf_params=None):
    def fitness_func(params):
        print("params: ", params)
        try:
            df = self.run(
                car_params={
                    "rho": 0.5,
                    "k_s": 10 ** params[0],
                    "k_c": 10 ** params[1],
                    "k_e": 10 ** params[2],
                    "k_r": 10 ** params[3],
                },
                cf_params=cf_params,
                has_ui=False, save_res=False
            )
        except Exception as e:
            print(f"Error: {e}")
            return 1e10
        ade = self.indicator_evaluation(df, opti=True)
        return ade

    def run_bayesian_opti():
        # 定义参数搜索空间
        space = [
            Real(-1, 1, name='k_s'),
            Real(-1, 1, name='k_c'),
            Real(-1, 1, name='k_e'),
            Real(-1, 1, name='k_r'),
        ]
        # x0 = list(itertools.product([0, 0.5, 1], [5, 10, 15, 20, 25]))
        # 运行贝叶斯优化
        res = gp_minimize(
            fitness_func,
            space,
            noise=0,
            acq_func="EI",  # Expected Improvement
            # x0=x0,  # 初始参数
            initial_point_generator="lhs",
            n_calls=10 * len(space),  # 总评估次数
            n_initial_points=10,  # 初始随机采样点
            verbose=True,
            random_state=2025,
        )
        print("最优参数:", res.x)
        print("最小目标值:", res.fun)
        return res

    result = run_bayesian_opti()

    res_dict = {}
    for i, (param, value) in enumerate(zip(result.space, result.x)):
        res_dict.update({param.name: 10 ** value})
    print(res_dict)

    return res_dict


if __name__ == '__main__':
    sim = make_road()
    run_road(sim)
    plot_data(sim, save_file_name=f"platoon_AV")
