# -*- coding: utf-8 -*-
# @Time : 2025/4/16 13:46
# @Author : yzbyx
# @File : scenario_loader.py
# Software: PyCharm
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

from traj_process.processor.map_phrase.map_config import US_101_Config, ExpresswayA_Config
from trasim_simplified.core.agent import Game_A_Vehicle, Game_O_Vehicle
from trasim_simplified.core.agent.game_agent import Game_Vehicle, Game_H_Vehicle
from trasim_simplified.core.constant import ScenarioTraj, V_TYPE, LCM, CFM, V_CLASS, COLOR, RouteType, ScenarioMode, \
    SurrClass
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.core.kinematics.cfm import get_cf_model
from trasim_simplified.util.calibrate.clb_cf_model import ga_cal, cf_param_ranges, cf_param_types, cf_param_ins, \
    show_traj
from trasim_simplified.util.scenario.scenario_util import make_road_from_osm, cal_dist_indicator
from trasim_simplified.core.constant import TrackInfo as C_Info
from trasim_simplified.util.scenario_plot import plot_scenario_twin
from trasim_simplified.util.tools import save_to_pickle, load_from_pickle

pd.options.mode.copy_on_write = True


class Scenario:
    def __init__(self, traj_data: ScenarioTraj, surr_class: SurrClass):
        self.weaving_offset = 60
        self.dataset_name = traj_data.dataset_name
        self.pattern_name = traj_data.pattern_name
        self.surr_class = surr_class
        self.traj_data = traj_data
        self.max_step = 1000
        self.surr_ids: dict[int, tuple[Game_Vehicle, pd.DataFrame, str]] = {}
        self.name_2_id = {}
        self.id_2_name = {}
        self.name_2_class = {}

        self.ev_type = None
        self.cp_type = None
        self.cr_type = None
        self.tr_type = None
        self.tp_type = None

        self.ev: Optional[Game_Vehicle] = None
        self.cp: Optional[Game_Vehicle] = None
        self.cpp: Optional[Game_Vehicle] = None
        self.cr: Optional[Game_Vehicle] = None
        self.crr: Optional[Game_Vehicle] = None
        self.tr: Optional[Game_Vehicle] = None
        self.trr: Optional[Game_Vehicle] = None
        self.tp: Optional[Game_Vehicle] = None
        self.tpp: Optional[Game_Vehicle] = None

        self.ev_rho = None
        self.tr_co = None
        self.tp_co = None
        self.rho_hat = []
        self.save_file_name = None
        self.lane_can_lc = None
        self.hv_cf_name = CFM.KK
        self.av_cf_name = CFM.TPACC
        self._load_road()
        self._clean_traj()

    def _load_road(self):
        if self.dataset_name == "NGSIM":
            osm = r"E:\BaiduSyncdisk\datasets\NGSIM\NGSIM_US_101.osm"
            map_config = US_101_Config["weaving1"]
        else:
            osm = r"E:\BaiduSyncdisk\datasets\CitySim\CitySim_ExpresswayA.osm"
            map_config = ExpresswayA_Config["weaving2"]
        self.fps = 10 if self.dataset_name == "NGSIM" else 30
        self.road: Road = make_road_from_osm(osm, map_config, self.weaving_offset)
        self.lanes = self.road.lane_list
        self._set_record_info()

    def _clean_traj(self):
        traj_list = [
            self.traj_data.EV_traj,
            self.traj_data.CP_traj, self.traj_data.CPP_traj,
            self.traj_data.TR_traj, self.traj_data.TRR_traj,
            self.traj_data.TP_traj, self.traj_data.TPP_traj,
            self.traj_data.OP_traj, self.traj_data.OPP_traj,
            self.traj_data.OR_traj, self.traj_data.ORR_traj,
            self.traj_data.CR_traj, self.traj_data.CRR_traj,
        ]
        traj_name = ["EV", "CP", "CPP", "TR", "TRR", "TP", "TPP", "OP", "OPP", "OR", "ORR", "CR", "CRR"]
        for i, (traj, name) in enumerate(zip(traj_list, traj_name)):
            if traj is None:
                continue
            if len(traj) == 0:
                setattr(self.traj_data, name + "_traj", None)
                continue
            track_id = traj["trackId"].values[0]

            self.name_2_id[name] = track_id
            self.id_2_name[track_id] = name

            if self.dataset_name != "NGSIM":
                traj = traj[traj["frame"] % 3 == 0]
                traj["frame"] = traj["frame"] // 3
            traj["frame"] -= traj["frame"].min()
            traj["name"] = name

            # 位置校正
            traj[C_Info.localLon] = traj["myLocalLon"].copy() + self.road.start_weaving_pos + self.weaving_offset
            # 根据每个轨迹点的车道id，获取车道的y_center，进而更新roadLat，即全局坐标
            traj[C_Info.localLat] = traj["myLocalLat"].copy()
            traj[C_Info.yCenterGlobal] = traj.apply(
                lambda row: self.lanes[row["laneId"]].y_center + row["myLocalLat"], axis=1
            )
            traj[C_Info.xCenterGlobal] = traj[C_Info.localLon].copy()  # 本仿真框架下，xCenterGlobal==localLon
            traj[C_Info.xFrontGlobal] = traj[C_Info.xCenterGlobal].copy() + traj["length"] / 2 * np.cos(traj["heading"])
            traj[C_Info.yFrontGlobal] = traj[C_Info.yCenterGlobal].copy() + traj["length"] / 2 * np.sin(traj["heading"])

            traj[C_Info.ttc] = traj["GapPC"] / traj["DvPC"]
            traj[C_Info.ttc] = traj[C_Info.ttc].apply(
                lambda x: np.inf if x < 0 else x
            )

            self.surr_ids.update({traj["trackId"].values[0]: (None, traj, name)})

    def _load_vehicle(self):
        self.road.reset()
        self._set_record_info()

        for i, (_, traj, name) in enumerate(self.surr_ids.values()):
            self.max_step = min(self.max_step, len(traj))
            length = traj["length"].values[0]
            x = traj[C_Info.xFrontGlobal].values[0]
            y = traj[C_Info.yFrontGlobal].values[0]
            lane_id = traj["laneId"].values[0]
            if name == "EV":
                self.lane_can_lc = [lane_id - 1, lane_id, lane_id + 1]
            heading = traj["heading"].values[0]
            speed = traj["speed"].values[0]
            if np.isnan(speed):
                speed = traj["speed"].values[1]
                assert not np.isnan(speed), f"speed is nan, {traj}"
            vel_max = np.nanmax(speed) + 5
            acc = traj["acceleration"].values[0]
            if np.isnan(acc):
                acc = traj["acceleration"].values[1]
                assert not np.isnan(acc), f"acceleration is nan, {traj}"
            vid = traj["trackId"].values[0]
            route_type = traj["routeClass"].values[0]

            if route_type == "mainline-mainline":
                route_type = RouteType.mainline
            elif route_type == "mainline-diverging":
                route_type = RouteType.diverge
            elif route_type == "merging-mainline":
                route_type = RouteType.merge
            elif route_type == "merging-diverging":
                route_type = RouteType.auxiliary
            else:
                raise ValueError(f"Unknown route type: {route_type}")

            car_class = self.surr_class.get_class(name)
            car = self._make_vehicle(
                lane_id, length, x, y, speed, acc, heading, vid, car_class, route_type
            )

            if name == "EV":
                self.ev = car
            elif name == "CP":
                self.cp = car
            elif name == "CR":
                self.cr = car
            elif name == "TR":
                self.tr = car
            elif name == "TP":
                self.tp = car

            car.cf_model.param_update({"v0": vel_max})

            self.surr_ids.update({vid: (car, traj, name)})

    def _make_vehicle(self, lane_id, length, global_x, global_y, speed, acc, heading, vid,
                      car_class: V_CLASS, route_type: RouteType):
        lane = self.lanes[lane_id]
        cf_name = self.hv_cf_name if car_class != V_CLASS.GAME_AV else self.av_cf_name
        destination_lanes = self.road.mainline_end_indexes \
            if route_type in [RouteType.mainline, RouteType.merge] else self.road.auxiliary_end_indexes
        if route_type == RouteType.mainline:
            color = COLOR.black
        elif route_type == RouteType.merge:
            color = COLOR.blue
        elif route_type == RouteType.diverge:
            color = COLOR.red
        else:
            color = COLOR.pink

        car = lane.car_insert(
            length, V_TYPE.PASSENGER, car_class,
            global_x, 0, 0,
            cf_name, {"v0": 30}, {"color": color},
            lc_name=LCM.MOBIL, lc_param={}, destination_lanes=destination_lanes, route_type=route_type
        )
        car.ID = vid
        car.y = global_y
        car.speed = speed
        car.acc = acc
        car.yaw = heading
        return car

    @staticmethod
    def set_vehicle_dynamic(car, global_x, global_y, speed, acc, heading):
        car.x = global_x
        car.y = global_y
        car.speed = speed
        car.acc = acc
        car.yaw = heading
        return car

    def run(
            self, cf_params: dict = None, car_params: dict = None,
            has_ui=True, save_res=True
    ):
        self.road.reset()
        self._load_vehicle()

        for vid, (veh, traj, name) in self.surr_ids.items():
            if isinstance(veh, Game_O_Vehicle):
                veh.skip = True
                veh.no_lc = True
            elif isinstance(veh, Game_H_Vehicle):
                if self.ev.route_type == RouteType.merge and veh.route_type == RouteType.diverge:
                    pass
                elif self.ev.route_type == RouteType.diverge and veh.route_type == RouteType.merge:
                    pass
                else:
                    veh.no_lc = True
            elif isinstance(veh, Game_A_Vehicle):
                if veh.ID == self.ev.ID:
                    veh.can_raise_game = True
                else:
                    veh.no_lc = True
            else:
                raise ValueError(f"Unknown vehicle class: {veh.__class__.__name__}")

        if cf_params is not None:
            if isinstance(list(cf_params.values())[0], dict):
                for vid, (car, traj, name) in self.surr_ids.items():
                    if vid in cf_params:
                        car.cf_model.param_update(cf_params[vid])
            else:
                self.ev.cf_model.param_update(cf_params)

        if car_params is not None:
            if isinstance(list(car_params.values())[0], dict):
                for vid, (car, traj, name) in self.surr_ids.items():
                    if vid in car_params:
                        car.set_car_param(car_params[vid])
            else:
                self.ev.set_car_param(car_params)

        TR_stra_dict = {}
        for step, stage in self.road.run(
                data_save=True, has_ui=has_ui, frame_rate=-1, warm_up_step=0,
                sim_step=self.max_step, dt=0.1
        ):
            if stage == 0 and step == 0 and has_ui:
                for vid, (car, traj, name) in self.surr_ids.items():
                    line = self.road.ui.ax.plot(traj[C_Info.xCenterGlobal].values,
                                                traj[C_Info.yCenterGlobal].values + self.road.lane_list[0].y_left,
                                                color='g', linewidth=1, alpha=0.3)[-1]
                    self.road.ui.static_item.append(line)

            if stage == 4 and has_ui:
                # 显示ev真实轨迹
                # self.road.ui.focus_on(ev)
                # self.road.ui.plot_hist_traj()
                # self.road.ui.plot_pred_traj()

                for veh in self.road.get_total_car():
                    if not isinstance(veh, Game_A_Vehicle):
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
                        # if veh.opti_game_res.TR_stra is not None:
                        #     tr_id = veh.opti_game_res.game_surr.TR.ID
                        #     if isinstance(veh.opti_game_res.game_surr.TR, Game_A_Vehicle):
                        #         rho = veh.opti_game_res.game_surr.TR.rho
                        #     else:
                        #         rho = np.mean(veh.rho_hat_s[tr_id])
                        #     TR_stra_dict[step] = \
                        #         (tr_id, veh.opti_game_res.TR_stra, rho)
                        # if veh.opti_game_res.TF_stra is not None:
                        #     tp_id = veh.opti_game_res.game_surr.TP.ID
                        #     if isinstance(veh.opti_game_res.game_surr.TP, Game_A_Vehicle):
                        #         rho = veh.opti_game_res.game_surr.TP.rho
                        #     else:
                        #         rho = np.mean(veh.rho_hat_s[tp_id])
                        #     TP_stra_dict[step] = \
                        #         (tp_id, veh.opti_game_res.TF_stra, rho)

                plt.pause(0.1)

            if stage == 4 and step < self.max_step:
                for vid, (car, traj, name) in self.surr_ids.items():
                    if isinstance(car, Game_O_Vehicle):
                        speed = traj["speed"].values[step]
                        x = traj[C_Info.xFrontGlobal].values[step]
                        y = traj[C_Info.yFrontGlobal].values[step]
                        heading = traj["heading"].values[step]
                        acc = traj["acceleration"].values[step]

                        self.set_vehicle_dynamic(car, x, y, speed, acc, heading)

        print("TR_stra_dict: ", TR_stra_dict)

        merged_df = self.get_res(save_res)
        return merged_df

    def _set_record_info(self):
        save_info = [C_Info.trackId, C_Info.frame, C_Info.time, C_Info.length, C_Info.width,
                     C_Info.xCenterGlobal, C_Info.yCenterGlobal, C_Info.xFrontGlobal, C_Info.yFrontGlobal,
                     C_Info.speed, C_Info.acc, C_Info.yaw, C_Info.delta,
                     C_Info.lane_add_num, C_Info.isLC, C_Info.ttc]
        lanes = self.road.lane_list
        lane_num = len(lanes)
        for i in range(lane_num):
            lanes[i].data_container.config(save_info=save_info, basic_info=False)

    def get_res(self, save_res=True):
        df = self.road.data_to_df()

        df[C_Info.localLonAcc] = df[C_Info.acc] * np.cos(df[C_Info.yaw])
        df[C_Info.localLatAcc] = df[C_Info.acc] * np.sin(df[C_Info.yaw])
        df[C_Info.localLonVel] = df[C_Info.speed] * np.cos(df[C_Info.yaw])
        df[C_Info.localLatVel] = df[C_Info.speed] * np.sin(df[C_Info.yaw])

        # 将原始traj拼接到df中
        df_ori = pd.DataFrame()
        for i, (car, traj, name) in self.surr_ids.items():
            temp = traj[["trackId", "frame", C_Info.xCenterGlobal, C_Info.yCenterGlobal,
                         C_Info.localLon, C_Info.localLat, C_Info.speed,
                         "laneId", "heading", "acceleration", "delta", C_Info.ttc]]
            temp.columns = [C_Info.trackId, C_Info.frame, C_Info.xCenterGlobal + "_ori", C_Info.yCenterGlobal + "_ori",
                            C_Info.localLon + "_ori", C_Info.localLat + "_ori", C_Info.speed + "_ori",
                            C_Info.laneId + "_ori", C_Info.yaw + "_ori", C_Info.acc + "_ori",
                            C_Info.delta + "_ori", C_Info.ttc + "_ori"]
            df_ori = pd.concat([df_ori, temp], axis=0)

        # 合并原始traj
        # 合并策略：保留后续(df)的列
        cols_from_merged = df.columns.difference(df_ori.columns).union(['trackId', 'frame'])
        merged_df = pd.merge(
            df[cols_from_merged],
            df_ori,
            on=['trackId', 'frame'],
            how='outer'
        )

        if save_res:
            save_to_pickle(
                merged_df,
                fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\data\{self.save_file_name}_sim_data.pkl"
            )

        return merged_df

    def _load_traj(self, merged_df=None):
        if merged_df is None:
            # 读取合并后的数据
            merged_df = load_from_pickle(
                fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\data\{self.save_file_name}_sim_data.pkl")
        traj_s = []
        traj_name = []
        for i in self.surr_ids.keys():
            traj = merged_df[merged_df[C_Info.trackId] == i].sort_values(C_Info.frame)
            traj_s.append(traj)
            traj_name.append(self.surr_ids[i][2])
        return merged_df, traj_s, traj_name

    def plot_scenario(self, highlight=None):
        if highlight is None:
            highlight = ["EV"]
        merged_df, traj_s, traj_name = self._load_traj()
        plot_scenario_twin(
            traj_s, traj_names=traj_name, road=self.road,
            fig_name=self.save_file_name, highlight=highlight,
        )

    def indicator_evaluation(self, merged_df=None):
        self._load_vehicle()
        df, traj_s, traj_name = self._load_traj(merged_df)

        ade_all = 0
        for name in traj_name:
            if self.surr_class.get_class(name) != V_CLASS.GAME_OV:
                traj = traj_s[traj_name.index(name)]
                ade, fde, lane_correct = cal_dist_indicator(traj)
                ade_all += ade * (5 if name == "EV" else 1)  # EV的权重为5，其余为1
                if not lane_correct:
                    ade_all += 1000  # 如果车道不正确，增加一个惩罚值

        return ade_all

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
                        'scale': params[4],
                        'preview_time': params[5],
                    },
                    cf_params=cf_params,
                    has_ui=False, save_res=False
                )
            except Exception as e:
                print(f"Error: {e}")
                return 1000
            ade = self.indicator_evaluation(df)
            return ade

        def run_bayesian_opti():
            # 定义参数搜索空间
            space = [
                Real(-1, 1, name='k_s'),
                Real(-1, 1, name='k_c'),
                Real(-1, 1, name='k_e'),
                Real(-1, 1, name='k_r'),
                Real(0, 1, name='scale'),
                Real(1, 10, name='preview_time'),
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

    def calibrate_cf(self, TR_AV=False, TP_AV=False):
        cf_param_dict_s = {}
        for vid, (car, traj, name) in self.surr_ids.items():
            if isinstance(car, Game_A_Vehicle):
                cf_name = self.av_cf_name
            elif isinstance(car, Game_H_Vehicle):
                cf_name = self.hv_cf_name
            else:
                continue
            cf_func = get_cf_model(cf_name)

            # try:
            track = traj
            evL = track["length"].values[0]
            track = track[track["myLeadId"] != -1]
            leaderL = track["myLeadLength"].values[0]

            obs_x = track["myLocalLon"].values + evL / 2 * np.cos(track["heading"].values)
            obs_v = track["myLonVelocity"].values
            obs_a = track["myLonAcceleration"].values
            obs_lx = track["myLeadLocalLon"].values + leaderL / 2
            obs_lv = track["myLeadLonVelocity"].values
            obs_la = track["myLeadLonAcceleration"].values

            results = ga_cal(
                cf_func=cf_func,
                obs_x=np.array(obs_x),
                obs_v=np.array(obs_v),
                obs_a=np.array(obs_a),
                obs_lx=np.array(obs_lx),
                obs_lv=np.array(obs_lv),
                obs_la=np.array(obs_la),
                leaderL=leaderL,
                dt=0.1,
                ranges=cf_param_ranges[cf_name],
                types=cf_param_types[cf_name],
                ins=cf_param_ins[cf_name],
                seed=2025, drawing=0, verbose=False
            )
            print(results["Vars"])
            # show_traj(
            #     cf_name, results["Vars"], 0.1,
            #     obs_x, obs_v, obs_a, obs_lx, obs_lv, obs_la, leaderL,
            #     traj_step=None, pair_ID=self.ev_id
            # )

            cf_param_dict_s.update({vid: results["Vars"]})
            # except Exception as e:
            #     print(f"Error: {e}")
            #     cf_param_dict_s.update({vid: {}})
            #     continue

        return cf_param_dict_s
