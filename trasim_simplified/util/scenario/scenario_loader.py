# -*- coding: utf-8 -*-
# @Time : 2025/4/16 13:46
# @Author : yzbyx
# @File : scenario_loader.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

from traj_process.processor.map_phrase.map_config import US_101_Config, ExpresswayA_Config
from trasim_simplified.core.agent import Vehicle, Game_H_Vehicle, Game_A_Vehicle
from trasim_simplified.core.agent.TwoDimSSM import TTC, traj_data_TTC
from trasim_simplified.core.agent.game_agent import Game_Vehicle
from trasim_simplified.core.constant import ScenarioTraj, V_TYPE, LCM, CFM, V_CLASS, COLOR, RouteType, ScenarioMode
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.core.kinematics.cfm import get_cf_model
from trasim_simplified.util.calibrate.clb_cf_model import ga_cal, cf_param_ranges, cf_param_types, cf_param_ins, \
    show_traj
from trasim_simplified.util.scenario.util import make_road_from_osm
from trasim_simplified.core.constant import TrackInfo as C_Info
from trasim_simplified.util.scenario_plot import plot_scenario, plot_scenario_twin
from trasim_simplified.util.tools import save_to_pickle, load_from_pickle

pd.options.mode.copy_on_write = True


class Scenario:
    def __init__(self, traj_data: ScenarioTraj, mode: str, ev_type=V_CLASS.GAME_HV):
        self.weaving_offset = 60
        self.dataset_name = traj_data.dataset_name
        self.pattern_name = traj_data.pattern_name
        self.traj_data = traj_data
        self.max_step = 1000
        self.ev_id = None
        self.tr_id = None
        self.tp_id = None
        self.cp_id = None
        self.cr_id = None
        self.surr_ids: dict[int, tuple[Game_Vehicle, pd.DataFrame, str]] = {}
        self.veh_name = {}
        self.mode = mode
        self.ev_type = ev_type
        self.ev_rho = None
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
            if name == "EV":
                self.ev_id = track_id
            elif name == "TR":
                self.tr_id = track_id
            elif name == "CP":
                self.cp_id = track_id
            elif name == "CR":
                self.cr_id = track_id
            elif name == "TP":
                self.tp_id = track_id
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

    def load_vehicle(self, ev_type=V_CLASS.GAME_HV):
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

            if i == 0:
                self.ev_id = vid
                car = self._make_vehicle(lane_id, length, x, y, speed, acc, heading, vid, ev_type, route_type)
            else:
                car = self._make_vehicle(lane_id, length, x, y, speed, acc, heading, vid, V_CLASS.GAME_HV, route_type)

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
            self, mode: ScenarioMode = None, cf_params: dict = None, car_params: dict = None,
            has_ui=True, save_res=True
    ):
        self.load_vehicle(self.ev_type)

        if mode is not None:
            self.mode = mode
        if self.mode == ScenarioMode.NO_INTERACTION:
            for veh, _, _ in self.surr_ids.values():
                if veh.ID != self.ev_id:
                    veh.skip = True
                veh.lane_can_lc = self.lane_can_lc
        else:
            for veh, _, _ in self.surr_ids.values():
                if veh.ID != self.ev_id:
                    veh.no_lc = True
                veh.lane_can_lc = self.lane_can_lc

        ev: Game_Vehicle = self.surr_ids[self.ev_id][0]
        ev.set_car_param({"color": COLOR.green})

        if cf_params is not None:
            if isinstance(list(cf_params.values())[0], dict):
                for vid, (car, traj, name) in self.surr_ids.items():
                    if vid in cf_params:
                        car.cf_model.param_update(cf_params[vid])
            else:
                ev.cf_model.param_update(cf_params)

        if car_params is not None:
            if isinstance(list(car_params.values())[0], dict):
                for vid, (car, traj, name) in self.surr_ids.items():
                    if vid in car_params:
                        car.set_car_param(car_params[vid])
            else:
                ev.set_car_param(car_params)

        if self.mode in [
            ScenarioMode.INTERACTION_TR_HV_TP_HV,
            ScenarioMode.INTERACTION_TR_HV_TP_AV
        ]:
            raise NotImplementedError("Interaction mode is not implemented yet.")
        elif self.mode == ScenarioMode.NO_INTERACTION:
            self.ev_rho = ev.rho
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.get_save_file_name(self.ev_rho)

        TR_stra_dict = {}
        for step, stage in self.road.run(
                data_save=True, has_ui=has_ui, frame_rate=-1, warm_up_step=0,
                sim_step=self.max_step, dt=0.1
        ):
            if stage == 0 and step == 0 and has_ui:
                for vid, (car, traj, name) in self.surr_ids.items():
                    line = self.road.ui.ax.plot(traj[C_Info.xCenterGlobal].values,
                                                traj[C_Info.yCenterGlobal].values + self.road.lane_list[0].y_left,
                                                color='g', linewidth=1, alpha=0.5)[-1]
                    self.road.ui.static_item.append(line)

            if stage == 4 and has_ui:
                # 显示ev真实轨迹
                self.road.ui.focus_on(ev)
                self.road.ui.plot_hist_traj()
                self.road.ui.plot_pred_traj()

                print("-" * 10 + "basic_info" + "-" * 10)
                print("step:", step, ev)
                print("-" * 10 + "gap_res_list" + "-" * 10)
                for res in ev.gap_res_list:
                    print(res)
                if ev.opti_gap is not None:
                    print("-" * 10 + "opti_gap_res" + "-" * 10)
                    print(ev.opti_gap)

                if isinstance(ev, Game_A_Vehicle) and ev.game_res_list is not None:
                    print("-" * 10 + "game_res_list" + "-" * 10)
                    for res in ev.game_res_list:
                        print(res)

                if ev.opti_game_res is not None:
                    print("-" * 10 + "opti_game_res" + "-" * 10)
                    print(ev.opti_game_res)
                    if ev.opti_game_res.TR_stra is not None:
                        tr_id = ev.opti_game_res.TR.ID
                        TR_stra_dict[step] =\
                            (tr_id, ev.opti_game_res.TR_stra, ev.rho_hat_s[tr_id])

                plt.pause(0.1)

            if stage == 4:
                if self.mode == ScenarioMode.NO_INTERACTION:
                    for vid, (car, traj, name) in self.surr_ids.items():
                        if vid != self.ev_id:
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
                     C_Info.xCenterGlobal, C_Info.yCenterGlobal,
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
                fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\data\{self.save_file_name}.pkl"
            )

        return merged_df

    def _load_traj(self, merged_df=None):
        if merged_df is None:
            # 读取合并后的数据
            merged_df = load_from_pickle(
                fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\data\{self.save_file_name}.pkl")
        traj_s = []
        traj_name = []
        for i in self.surr_ids.keys():
            traj = merged_df[merged_df[C_Info.trackId] == i].sort_values(C_Info.frame)
            traj_s.append(traj)
            traj_name.append(self.surr_ids[i][2])
        return merged_df, traj_s, traj_name

    def get_save_file_name(self, rho=None):
        if rho is not None:
            self.ev_rho = rho
        if self.mode == ScenarioMode.NO_INTERACTION:
            self.save_file_name = \
                (fr"{self.dataset_name}_{self.pattern_name}_{self.traj_data.track_id}_"
                 fr"{self.mode}_EV-{self.ev_rho}")
        elif self.mode == ScenarioMode.INTERACTION_TR_HV_TP_HV:
            self.save_file_name = \
                (fr"{self.dataset_name}_{self.pattern_name}_{self.traj_data.track_id}_"
                 fr"{self.mode}_HV-{self.ev_rho}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self.save_file_name

    def plot_scenario(self, rho=None):
        self.get_save_file_name(rho)
        merged_df, traj_s, traj_name = self._load_traj()
        plot_scenario_twin(
            traj_s, traj_names=traj_name, road=self.road,
            fig_name=self.save_file_name
        )

    def indicator_evaluation(self, merged_df=None, opti=False):
        df, traj_s, traj_name = self._load_traj(merged_df)

        # 行驶距离偏差
        x_center = traj_s[0][C_Info.xCenterGlobal].to_numpy()
        y_center = traj_s[0][C_Info.yCenterGlobal].to_numpy()
        x_center_ori = traj_s[0][C_Info.xCenterGlobal + "_ori"].to_numpy()
        y_center_ori = traj_s[0][C_Info.yCenterGlobal + "_ori"].to_numpy()
        nan_indexes = np.isnan(x_center) | np.isnan(y_center) | np.isnan(x_center_ori) | np.isnan(y_center_ori)
        x_center = x_center[~nan_indexes]
        y_center = y_center[~nan_indexes]
        x_center_ori = x_center_ori[~nan_indexes]
        y_center_ori = y_center_ori[~nan_indexes]
        # x_dist = np.max(x_center) - np.min(x_center)
        # y_dist = np.max(y_center) - np.min(y_center)
        # x_e = np.sqrt((x_center - x_center_ori) ** 2) / x_dist
        # y_e = np.sqrt((y_center - y_center_ori) ** 2) / y_dist
        distance = np.linalg.norm(
            np.array([x_center - x_center_ori, y_center - y_center_ori]), axis=0
        )
        ade = np.mean(distance)

        # 计算车道变更
        ev_traj = traj_s[traj_name.index("EV")]
        lane_end = ev_traj[C_Info.laneId].values[-1]
        lane_end_ori = ev_traj[C_Info.laneId + "_ori"].values[-2]  # 仿真会多记一次

        if opti:
            if lane_end != lane_end_ori:
                return 1e10
            return ade

        print("车道变更：", lane_end == lane_end_ori)

        fde = distance[-1]
        print("ade: ", ade, "fde: ", fde)

        # 计算指标
        # 计算速度偏差
        v = traj_s[0][C_Info.speed].to_numpy()
        v_ori = traj_s[0][C_Info.speed + "_ori"].to_numpy()
        v_error = np.abs(v - v_ori)
        v_error = v_error[~np.isnan(v_error)]
        mean_v_error = np.mean(v_error)

        # 计算加速度偏差
        a = traj_s[0][C_Info.acc].to_numpy()
        a_ori = traj_s[0][C_Info.acc + "_ori"].to_numpy()
        a_error = np.abs(a - a_ori)
        a_error = a_error[~np.isnan(a_error)]
        mean_a_error = np.mean(a_error)
        print("ave: ", mean_v_error, "aae: ", mean_a_error)

        # 安全2D-TTC

        # min_ttc = np.inf
        # min_ttc_ori = np.inf
        # for name in ["CP", "TP", "TR", "CR"]:
        #     if name not in traj_name:
        #         continue
        #     traj = traj_s[traj_name.index(name)]
        #     ttc_seq = traj_data_TTC(ev_traj, traj, is_ori=False)
        #     ttc_seq_ori = traj_data_TTC(ev_traj, traj, is_ori=True)
        #     min_ttc = min(np.nanmin(ttc_seq), min_ttc)
        #     min_ttc_ori = min(np.nanmin(ttc_seq_ori), min_ttc_ori)
        # print("min_ttc: ", min_ttc, "min_ttc_ori: ", min_ttc_ori)

        min_ttc = min(ev_traj[C_Info.ttc])
        min_ttc_ori = min(ev_traj[C_Info.ttc + "_ori"])
        print("安全提升：", min_ttc - min_ttc_ori)

        # 效率
        eff = np.nanmean(v) - v[0]
        eff_ori = np.nanmean(v_ori) - v_ori[0]
        print("效率提升：", eff - eff_ori)

        # 舒适性
        yaw = ev_traj[C_Info.yaw].to_numpy()
        yaw_ori = ev_traj[C_Info.yaw + "_ori"].to_numpy()
        ax_max = np.nanmax(np.abs(a * np.sin(yaw)))
        ay_max = np.nanmax(np.abs(a * np.cos(yaw)))
        ax_ori_max = np.nanmax(np.abs(a_ori * np.sin(yaw_ori)))
        ay_ori_max = np.nanmax(np.abs(a_ori * np.cos(yaw_ori)))
        # print("ax_max: ", ax_max, "ay_max: ", ay_max, "ax_ori_max: ",
        #       ax_ori_max, "ay_ori_max: ", ay_ori_max, "delta com cost: ", ax_max)
        print("舒适提升：", ax_ori_max + ay_ori_max - ax_max - ay_max)

        # ev: Game_Vehicle = self.surr_ids[self.ev_id][0]
        # ev.cal_cost_by_traj()

        # d_yaw_max = np.nanmax(np.diff(yaw) / 0.1)
        # d_yaw_max_ori = np.nanmax(np.diff(yaw_ori) / 0.1)
        # print("d_yaw_max: ", d_yaw_max, "d_yaw_max_ori: ", d_yaw_max_ori)

        return ade

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

    def calibrate_cf(self, TR_AV=False, TP_AV=False):
        cf_param_dict_s = {}
        for vid, (car, traj, name) in self.surr_ids.items():
            if vid == self.ev_id:
                cf_name = self.av_cf_name
            else:
                if name == "TR" and TR_AV:
                    cf_name = self.av_cf_name
                elif name == "TP" and TP_AV:
                    cf_name = self.av_cf_name
                else:
                    cf_name = self.hv_cf_name
            cf_func = get_cf_model(cf_name)

            try:
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
            except Exception as e:
                print(f"Error: {e}")
                cf_param_dict_s.update({vid: {}})
                continue

        return cf_param_dict_s
