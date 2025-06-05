# -*- coding: utf-8 -*-
# @Time : 2025/4/21 9:31
# @Author : yzbyx
# @File : scenario_TR_interaction.py
# Software: PyCharm
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

np.random.seed(2025)

from trasim_simplified.core.agent.game_agent import Game_Vehicle, Game_A_Vehicle
from trasim_simplified.core.constant import ScenarioMode, COLOR, ScenarioTraj, V_CLASS, RouteType
from trasim_simplified.util.scenario.scenario_loader import Scenario
from trasim_simplified.core.constant import TrackInfo as C_Info
from trasim_simplified.util.scenario_plot import plot_scenario_twin, plot_stra, plot_planned_traj
from trasim_simplified.util.tools import load_from_pickle, save_to_pickle


class ScenarioInteractionCo(Scenario):
    def __init__(self, traj_data: ScenarioTraj, mode: str,
                 ev_type: V_CLASS = V_CLASS.GAME_HV,
                 tr_type: V_CLASS = V_CLASS.GAME_HV,
                 tp_type: V_CLASS = V_CLASS.GAME_HV):
        super().__init__(traj_data, mode, ev_type)
        self.tr_rho = None
        self.tr_stra_dict = None
        self.tp_stra_dict = None
        self.tr_type = tr_type
        self.tp_type = tp_type

    def _load_vehicle(self, ev_type: str = V_CLASS.GAME_AV):
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
                if vid == self.tr_id:
                    v_class = self.tr_type
                elif vid == self.tp_id:
                    v_class = self.tp_type
                else:
                    v_class = V_CLASS.GAME_HV
                car = self._make_vehicle(lane_id, length, x, y, speed, acc, heading, vid, v_class, route_type)

            car.cf_model.param_update({"v0": vel_max})

            self.surr_ids.update({vid: (car, traj, name)})

    def run(self, mode: ScenarioMode = None, cf_params: dict = None, car_params: dict = None,
            has_ui=True, save_res=True, ev_co=None, tr_co=None, tp_co=None):
        self._load_vehicle(self.ev_type)

        if mode is not None:
            self.mode = mode
        for veh, _, _ in self.surr_ids.values():
            if veh.ID in [self.ev_id]:
                # veh.lc_incentive = True
                veh.lc_incentive = False
            else:
                veh.no_lc = True
            veh.lane_can_lc = self.lane_can_lc

        ev: Game_Vehicle = self.surr_ids[self.ev_id][0]
        ev.set_car_param({"color": COLOR.green})

        if cf_params is not None:
            for vid, (car, traj, name) in self.surr_ids.items():
                if vid in cf_params:
                    car.cf_model.param_update(cf_params[vid])

        if car_params is not None:
            for vid, (car, traj, name) in self.surr_ids.items():
                car.set_car_param(car_params)

        self.surr_ids[self.tr_id][0].set_car_param({"game_co": self.tr_co})
        self.surr_ids[self.tp_id][0].set_car_param({"game_co": self.tp_co})

        self.get_save_file_name(self.ev_rho)

        # lane_end_ori = self.traj_data.EV_traj["laneId"].values[-1]
        # ev = self.surr_ids[self.ev_id][0].la

        TR_stra_dict = {}
        TP_stra_dict = {}
        EV_planned_traj = []
        for step, stage in self.road.run(
                data_save=True, has_ui=has_ui, frame_rate=-1, warm_up_step=0,
                sim_step=self.max_step, dt=0.1
        ):
            if stage == 0 and step == 0 and has_ui:
                for vid, (car, traj, name) in self.surr_ids.items():
                    line = self.road.ui.ax.plot(
                        traj[C_Info.xCenterGlobal].values,
                        traj[C_Info.yCenterGlobal].values + self.road.lane_list[0].y_left,
                        color='g', linewidth=1, alpha=0.5
                    )[-1]
                    self.road.ui.static_item.append(line)

            if stage == 4 and has_ui:
                # 显示ev真实轨迹
                self.road.ui.focus_on(ev)
                self.road.ui.plot_hist_traj()
                self.road.ui.plot_pred_traj()

                print("-" * 10 + "basic_info" + "-" * 10)
                print("step:", step, ev)

                if ev.gap_res_list is not None:
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
                        if isinstance(ev.opti_game_res.TR, Game_A_Vehicle):
                            rho = ev.opti_game_res.TR.rho
                        else:
                            rho = np.mean(ev.rho_hat_s[tr_id])
                        TR_stra_dict[step] = \
                            (tr_id, ev.opti_game_res.TR_stra, rho)
                    if ev.lc_conti_time == ev.lane.dt:
                        EV_planned_traj.append(ev.opti_game_res.EV_opti_traj)
                    if ev.opti_game_res.TF_stra is not None:
                        tp_id = ev.opti_game_res.TF.ID
                        TP_stra_dict[step] = \
                            (tp_id, ev.opti_game_res.TF_stra, ev.opti_game_res.TF.rho)

                plt.pause(0.1)

            if stage == 4:
                for vid, (car, traj, name) in self.surr_ids.items():
                    if vid not in [
                        self.ev_id,
                        # self.tr_id, self.cp_id,
                        # self.cr_id, self.trr_id, self.tp_id, self.tpp_id
                    ]:
                        speed = traj["speed"].values[step]
                        x = traj[C_Info.xFrontGlobal].values[step]
                        y = traj[C_Info.yFrontGlobal].values[step]
                        heading = traj["heading"].values[step]
                        acc = traj["acceleration"].values[step]

                        self.set_vehicle_dynamic(car, x, y, speed, acc, heading)

        print("TR_stra_dict: ", TR_stra_dict)
        self.tr_stra_dict = TR_stra_dict

        # 保存数据
        save_to_pickle(
            self.tr_stra_dict,
            fr"{self.save_file_name}_TR_stra.pkl"
        )

        save_to_pickle(
            self.tp_stra_dict,
            fr"{self.save_file_name}_TP_stra.pkl"
        )

        save_to_pickle(
            EV_planned_traj,
            fr"{self.save_file_name}_EV_planned_traj.pkl"
        )

        merged_df = self.get_res(save_res)
        return merged_df

    def get_save_file_name(self, tr_co=None, tp_co=None):
        if tr_co is not None:
            self.tr_co = tr_co
        if tp_co is not None:
            self.tp_co = tp_co

        self.save_file_name = \
            (fr"{self.dataset_name}_{self.pattern_name}_{self.traj_data.track_id}_"
             fr"{self.mode}_TR-{self.tr_co}_TP-{self.tp_co}")
        return self.save_file_name

    def _load_stra_data(self):
        """
        加载策略数据
        :return: 策略数据
        """
        data = load_from_pickle(
            fr"{self.save_file_name}_TR_stra.pkl"
        )
        try:
            tp_data = load_from_pickle(
                fr"{self.save_file_name}_TP_stra.pkl"
            )
        except:
            tp_data = None
        return data, tp_data

    def _load_game_planned_traj(self):
        data = load_from_pickle(
            fr"{self.save_file_name}_EV_planned_traj.pkl"
        )
        return data

    def plot_scenario(self, tr_co=None, tp_co=None, highlight=None):
        if highlight is None:
            highlight = ["EV", "TR", "TP"]
        if tr_co is not None:
            self.tr_co = tr_co
        if tp_co is not None:
            self.tp_co = tp_co

        self.get_save_file_name(tr_co=self.tr_co, tp_co=self.tp_co)
        merged_df, traj_s, traj_name = self._load_traj()
        tr_stra_dict, tp_stra_dict = self._load_stra_data()
        game_planned_traj = self._load_game_planned_traj()

        plot_stra(tr_stra_dict, self.max_step, self.save_file_name, "TR")
        if isinstance(self.surr_ids[self.tp_id][0], Game_A_Vehicle):
            plot_stra(tp_stra_dict, self.max_step, self.save_file_name, "TP")

        plot_scenario_twin(
            traj_s, traj_names=traj_name, road=self.road,
            fig_name=self.save_file_name, mode=self.mode, highlight=highlight
        )

        ev_traj = traj_s[traj_name.index("EV")]
        plot_planned_traj(game_planned_traj, self.save_file_name, self.road, ev_traj)


if __name__ == '__main__':
    scenario_path = r"E:\BaiduSyncdisk\weaving-analysis\data\pattern_scenario_data.pkl"
    scenario_data: dict[str, list[ScenarioTraj]] = load_from_pickle(scenario_path)  # pattern_name, pattern_traj_s

    for pattern_name, pattern_traj_s in scenario_data.items():
        have_one = False
        for pattern_traj in pattern_traj_s:
            name = f"{pattern_traj.dataset_name}_{pattern_name}_{pattern_traj.track_id}"
            if name in [
                "CitySim_驶出_2717", "CitySim_驶出_14385", "NGSIM_驶入_1257",
                "CitySim_驶入_959", "CitySim_驶入_5053", "CitySim_驶入_2014",
                "CitySim_驶入_4782", "CitySim_驶入_955", "CitySim_驶入_2130",
                "CitySim_驶入_20307", "NGSIM_预驶出_1547", "CitySim_预驶出_2343",
                "CitySim_预驶出_2141", "CitySim_松弛行为_3122", "CitySim_松弛行为_21140",
                "CitySim_预期行为_778"
            ]:
                continue
            # if pattern_name not in ["驶入", "驶出"]:
            #     continue

            if pattern_traj.CP_traj is None:
                continue
            if pattern_traj.TR_traj is None or pattern_traj.TP_traj is None:
                continue
            if len(pattern_traj.TR_traj) > 70 and pattern_traj.dataset_name == "NGSIM":
                continue
            if len(pattern_traj.TP_traj) > 70 * 3 and pattern_traj.dataset_name == "CitySim":
                continue
            if (pattern_traj.EV_traj["myLocalLon"].values[0] -
                    pattern_traj.TR_traj["myLocalLon"].values[0] > 20):
                continue
            if (pattern_traj.EV_traj["myLocalLon"].values[0] -
                    pattern_traj.TP_traj["myLocalLon"].values[0] < -20):
                continue
            print(name, "tr_id", pattern_traj.TR_traj["trackId"].values[0],
                  "tp_id", pattern_traj.TP_traj["trackId"].values[0])
            base_path = r"E:\BaiduSyncdisk\car-following-model\tests\thesis\data"

            tr_av = False
            tp_av = False
            tr_co = tp_co = 0.5

            if not tr_av and not tp_av:
                sce = ScenarioInteractionCo(
                    pattern_traj,
                    ScenarioMode.INTERACTION_TR_HV_TP_HV,
                    ev_type=V_CLASS.GAME_AV,
                    tr_type=V_CLASS.GAME_HV,
                    tp_type=V_CLASS.GAME_HV
                )
            elif tr_av and not tp_av:
                sce = ScenarioInteractionCo(
                    pattern_traj,
                    ScenarioMode.INTERACTION_TR_AV_TP_HV,
                    ev_type=V_CLASS.GAME_AV,
                    tr_type=V_CLASS.GAME_AV,
                    tp_type=V_CLASS.GAME_HV
                )
            elif not tr_av and tp_av:
                sce = ScenarioInteractionCo(
                    pattern_traj,
                    ScenarioMode.INTERACTION_TR_HV_TP_AV,
                    ev_type=V_CLASS.GAME_AV,
                    tr_type=V_CLASS.GAME_HV,
                    tp_type=V_CLASS.GAME_AV
                )
            else:
                sce = ScenarioInteractionCo(
                    pattern_traj,
                    ScenarioMode.INTERACTION_TR_AV_TP_AV,
                    ev_type=V_CLASS.GAME_AV,
                    tr_type=V_CLASS.GAME_AV,
                    tp_type=V_CLASS.GAME_AV
                )

            cf_params = {}
            if not os.path.exists(fr"{base_path}\{name}_cf_params_TR-AV-{tr_av}_TP-AV-{tp_av}.pkl"):
                cf_params = sce.calibrate_cf(TR_AV=tr_av, TP_AV=tp_av)
                save_to_pickle(
                    cf_params,
                    fr"{base_path}\{name}_cf_params_TR-AV-{tr_av}_TP-AV-{tp_av}.pkl"
                )
            cf_params = load_from_pickle(fr"{base_path}\{name}_cf_params_TR-AV-{tr_av}_TP-AV-{tp_av}.pkl")
            print(cf_params)

            car_params = {}
            # if not os.path.exists(fr"{base_path}\{name}_car_params.pkl"):
            #     car_params = sce.opti_ade(cf_params)
            #     save_to_pickle(
            #         car_params,
            #         fr"{base_path}\{name}_car_params.pkl"
            #     )
            # car_params = load_from_pickle(fr"{base_path}\{name}_car_params.pkl")
            # print(car_params)

            save_file_name = sce.get_save_file_name(tr_co=tr_co, tp_co=tp_co)
            print(save_file_name)
            if not os.path.exists(fr"{base_path}\{save_file_name}.pkl"):
                sce.run(cf_params=cf_params, car_params=car_params,
                        has_ui=True, tr_co=tr_co, tp_co=tp_co)
            sce.plot_scenario(tr_co=tr_co, tp_co=tp_co, highlight=["EV"])
            sce.indicator_evaluation()

            have_one = True
            break
        if have_one:
            break
