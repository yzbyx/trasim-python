# -*- coding: utf-8 -*-
# @Time : 2025/4/21 9:31
# @Author : yzbyx
# @File : scenario_TR_interaction.py
# Software: PyCharm
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from trasim_simplified.core.agent.game_agent import Game_Vehicle, Game_A_Vehicle
from trasim_simplified.core.constant import ScenarioMode, COLOR, ScenarioTraj, V_CLASS
from trasim_simplified.util.scenario.scenario_loader import Scenario
from trasim_simplified.core.constant import TrackInfo as C_Info
from trasim_simplified.util.scenario_plot import plot_scenario_twin
from trasim_simplified.util.tools import load_from_pickle, save_to_pickle

np.random.seed(2025)


class ScenarioTRInteraction(Scenario):
    def __init__(self, traj_data: ScenarioTraj, mode: str, ev_type: V_CLASS = V_CLASS.GAME_HV):
        super().__init__(traj_data, mode, ev_type)
        self.tr_rho = None
        self.tr_stra_dict = None

    def run(self, mode: ScenarioMode = None, cf_params: dict = None, car_params: dict = None,
            has_ui=True, save_res=True, ev_rho=None, tr_rho=None):
        self.load_vehicle(self.ev_type)

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

        if ev_rho is not None:
            ev.rho = ev_rho
        if tr_rho is not None:
            for vid, (car, traj, name) in self.surr_ids.items():
                if vid == self.tr_id:
                    car.rho = tr_rho

        self.ev_rho = ev.rho
        self.tr_rho = tr_rho

        self.get_save_file_name(self.ev_rho)

        # lane_end_ori = self.traj_data.EV_traj["laneId"].values[-1]
        # ev = self.surr_ids[self.ev_id][0].la

        TR_stra_dict = {}
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
                        tr_id = ev.opti_game_res.game_surr.TR.ID
                        TR_stra_dict[step] = \
                            (tr_id, ev.opti_game_res.TR_stra, np.mean(ev.rho_hat_s[tr_id]))
                    if ev.lc_conti_time == ev.lane.dt:
                        EV_planned_traj.append(ev.opti_game_res.EV_opti_traj)

                plt.pause(0.1)

            # if stage == 4:
            #     for vid, (car, traj, name) in self.surr_ids.items():
            #         if vid not in [self.ev_id, self.tr_id, self.cp_id, self.cr_id]:
            #             speed = traj["speed"].values[step]
            #             x = traj[C_Info.xFrontGlobal].values[step]
            #             y = traj[C_Info.yFrontGlobal].values[step]
            #             heading = traj["heading"].values[step]
            #             acc = traj["acceleration"].values[step]
            #
            #             self.set_vehicle_dynamic(car, x, y, speed, acc, heading)

        print("TR_stra_dict: ", TR_stra_dict)
        self.tr_stra_dict = TR_stra_dict
        self.tp_stra_dict = TP_stra_dict

        # 保存数据
        save_to_pickle(
            self.tr_stra_dict,
            fr"{self.save_file_name}_TR_stra.pkl"
        )

        save_to_pickle(
            EV_planned_traj,
            fr"{self.save_file_name}_EV_planned_traj.pkl"
        )

        merged_df = self.get_res(save_res)
        return merged_df

    def get_save_file_name(self, ev_rho=None, tr_rho=None):
        if ev_rho is not None:
            self.ev_rho = ev_rho
        if tr_rho is not None:
            self.tr_rho = tr_rho

        self.save_file_name = \
            (fr"{self.dataset_name}_{self.pattern_name}_{self.traj_data.track_id}_"
             fr"{self.mode}_EV-{self.ev_rho}_TR-{self.tr_rho}")
        return self.save_file_name

    def _load_stra_data(self):
        """
        加载策略数据
        :return: 策略数据
        """
        data = load_from_pickle(
            fr"{self.save_file_name}_TR_stra.pkl"
        )
        return data

    def _load_game_planned_traj(self):
        data = load_from_pickle(
            fr"{self.save_file_name}_EV_planned_traj.pkl"
        )
        return data

    def plot_scenario(self, ev_rho=None, tr_rho=None):
        if ev_rho is not None:
            self.ev_rho = ev_rho
        if tr_rho is not None:
            self.tr_rho = tr_rho

        self.get_save_file_name(self.ev_rho, self.tr_rho)
        merged_df, traj_s, traj_name = self._load_traj()
        tr_stra_dict = self._load_stra_data()
        tp_stra_dict = self.tp_stra_dict
        game_planned_traj = self._load_game_planned_traj()

        # 绘制激进度估计与TR策略变化曲线
        tr_id_s = []
        tr_stra_s = []
        tr_rho_s = []
        if tr_stra_dict is not None:
            steps = np.arange(self.max_step)
            for step in steps:
                if step in tr_stra_dict:
                    tr_id, tr_stra, rho = tr_stra_dict[step]
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
            _width = 70 * mm  # 图片宽度英寸
            _ratio = 5 / 7  # 图片长宽比
            figsize = (_width, _width * _ratio)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            fig: plt.Figure = fig
            ax: plt.Axes = ax
            # 根据tr_id的数量对stra和rho进行绘制
            tr_ids_unique = np.unique(tr_id_s)
            tr_ids_unique = tr_ids_unique[~np.isnan(tr_ids_unique) & (tr_ids_unique > 0)]
            colors = seaborn.color_palette('Set1', n_colors=len(tr_ids_unique))
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
                fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{self.save_file_name}_TR_stra.png",
                dpi=300, bbox_inches='tight'
            )

        plot_scenario_twin(
            traj_s, traj_names=traj_name, road=self.road,
            fig_name=self.save_file_name, mode=self.mode
        )

        num = len(game_planned_traj)
        if num > 0:
            mm = 1 / 25.4  # mm转inch
            _width = 70 * mm * 2  # 图片宽度英寸
            _ratio = 5 / 14  # 图片长宽比
            figsize = (_width, _width * _ratio)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            fig: plt.Figure = fig
            ax: plt.Axes = ax
            self.road.draw(ax=ax, fill=False)

            min_x = None
            max_x = None
            min_y = None
            max_y = None
            for i, traj in enumerate(game_planned_traj):
                ax.plot(traj[:, 0], traj[:, 3], color="g",
                        alpha=(i + 1) * 1 / num, linewidth=1)
                # 端点绘制
                ax.scatter(
                    traj[0, 0], traj[0, 3], color="orange", s=10, alpha=(i + 1) * 1 / num
                )
                ax.scatter(
                    traj[-1, 0], traj[-1, 3], color="blue", s=10, alpha=(i + 1) * 1 / num
                )
                ax.text(
                    traj[-1, 0], traj[-1, 3], str(i + 1), color="k", fontsize=10,
                    ha='center', va='bottom'
                )
                min_x = min(traj[:, 0]) if min_x is None else min(min_x, np.min(traj[:, 0]))
                max_x = max(traj[:, 0]) if max_x is None else max(max_x, np.max(traj[:, 0]))
                min_y = min(traj[:, 3]) if min_y is None else min(min_y, np.min(traj[:, 3]))
                max_y = max(traj[:, 3]) if max_y is None else max(max_y, np.max(traj[:, 3]))

            ev_traj = traj_s[traj_name.index("EV")]
            ax.plot(
                ev_traj["xFrontGlobal"].values,
                ev_traj["yFrontGlobal"].values,
                color='r', linewidth=1, alpha=0.5
            )
            # 端点绘制
            ax.scatter(
                ev_traj["xFrontGlobal"].values[0],
                ev_traj["yFrontGlobal"].values[0],
                color="orange", s=20, alpha=1, marker="^"
            )
            ax.scatter(
                ev_traj["xFrontGlobal"].values[-1],
                ev_traj["yFrontGlobal"].values[-1],
                color="blue", s=20, alpha=1, marker="^"
            )

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_xlim(min_x - 10, max_x + 10)
            ax.set_ylim(min_y - 10, max_y + 10)

            fig.savefig(
                fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\fig\{self.save_file_name}_EV_planned_traj.png",
                dpi=300, bbox_inches='tight'
            )


if __name__ == '__main__':
    scenario_path = r"E:\BaiduSyncdisk\weaving-analysis\data\pattern_scenario_data.pkl"
    scenario_data: dict[str, list[ScenarioTraj]] = load_from_pickle(scenario_path)  # pattern_name, pattern_traj_s

    for pattern_name, pattern_traj_s in scenario_data.items():
        have_one = False
        for pattern_traj in pattern_traj_s:
            name = f"{pattern_traj.dataset_name}_{pattern_name}_{pattern_traj.track_id}"
            if name in [
                "CitySim_驶入_959", "CitySim_驶入_1353", "CitySim_驶入_2648", "CitySim_驶入_5053",
                "CitySim_驶入_6218", "NGSIM_预期行为_464", "NGSIM_预期行为_243", "NGSIM_交织换道_1111",
                "NGSIM_交织换道_1135", "CitySim_交织换道_955", "CitySim_交织换道_19860"
            ]:
                continue
            if pattern_name != "交织换道":
                continue
            # if name not in ["NGSIM_预期行为_243"]:
            #     continue

            # if pattern_traj.dataset_name == "NGSIM":
            #     continue
            if pattern_traj.TR_traj is None:
                continue
            if (pattern_traj.EV_traj["myLocalLon"].values[0] -
                    pattern_traj.TR_traj["myLocalLon"].values[0] > 20):
                continue
            print(name)
            base_path = r"E:\BaiduSyncdisk\car-following-model\tests\thesis\data"
            sce = ScenarioTRInteraction(
                pattern_traj,
                ScenarioMode.INTERACTION_TR_HV_TP_HV,
                ev_type=V_CLASS.GAME_AV
            )

            if not os.path.exists(fr"{base_path}\{name}_cf_params.pkl"):
                cf_params = sce.calibrate_cf()
                save_to_pickle(
                    cf_params,
                    fr"{base_path}\{name}_cf_params.pkl"
                )
            cf_params = load_from_pickle(fr"{base_path}\{name}_cf_params.pkl")
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

            ev_rho = 0.5
            for tr_rho in [0.5]:
                save_file_name = sce.get_save_file_name(ev_rho=ev_rho, tr_rho=tr_rho)
                print(save_file_name)
                if not os.path.exists(fr"{base_path}\{save_file_name}.pkl"):
                    sce.run(cf_params=cf_params, car_params=car_params,
                            has_ui=True, ev_rho=ev_rho, tr_rho=tr_rho)
                sce.plot_scenario(ev_rho=ev_rho, tr_rho=tr_rho)
                sce.indicator_evaluation()
            have_one = True
            break
        if have_one:
            break
