# -*- coding: utf-8 -*-
# @Time : 2025/4/16 15:33
# @Author : yzbyx
# @File : run_scenario_base.py
# Software: PyCharm
import os

import numpy as np

from trasim_simplified.core.constant import ScenarioTraj, ScenarioMode, V_CLASS
from trasim_simplified.util.scenario.scenario_loader import Scenario
from trasim_simplified.util.tools import load_from_pickle, save_to_pickle

if __name__ == '__main__':
    scenario_path = r"E:\BaiduSyncdisk\weaving-analysis\data\pattern_scenario_data.pkl"
    scenario_data: dict[str, list[ScenarioTraj]] = load_from_pickle(scenario_path)  # pattern_name, pattern_traj_s

    for pattern_name, pattern_traj_s in scenario_data.items():
        have_one = False
        for pattern_traj in pattern_traj_s:
            name = f"{pattern_traj.dataset_name}_{pattern_name}_{pattern_traj.track_id}"
            # if name in [
            #     "CitySim_驶入_959", "CitySim_驶入_1353", "CitySim_驶入_2648", "CitySim_驶入_5053",
            #     "CitySim_驶入_6218", "NGSIM_预期行为_464", "NGSIM_预期行为_243", "NGSIM_预期行为_1191"
            # ]:
            #     continue
            # if name not in ["NGSIM_预期行为_243"]:
            #     continue
            # if name not in ["CitySim_驶入_207"]:
            #     continue
            if name not in ["NGSIM_驶出_1706"]:
                continue
            # print("find")
            # if pattern_traj.dataset_name == "NGSIM":
            #     continue
            # if pattern_name != "交织换道":
            #     continue
            # if name not in ["NGSIM_避让换道_1030"]:
            #     continue
            print(name)
            base_path = r"E:\BaiduSyncdisk\car-following-model\tests\thesis\data"
            sce = Scenario(pattern_traj, ScenarioMode.NO_INTERACTION, ev_type=V_CLASS.GAME_AV)

            if not os.path.exists(fr"{base_path}\{name}_cf_params.pkl"):
                cf_params = sce.calibrate_cf()
                save_to_pickle(
                    cf_params,
                    fr"{base_path}\{name}_cf_params.pkl"
                )
            cf_params = load_from_pickle(fr"{base_path}\{name}_cf_params.pkl")

            car_params = {}
            # if not os.path.exists(fr"{base_path}\{name}_car_params.pkl"):
            #     car_params = sce.opti_ade(cf_params)
            #     save_to_pickle(
            #         car_params,
            #         fr"{base_path}\{name}_car_params.pkl"
            #     )
            # car_params = load_from_pickle(fr"{base_path}\{name}_car_params.pkl")

            for rho in [0.5, 0.1, 0.9]:
                car_params["rho"] = rho
                save_file_name = sce.get_save_file_name(rho)
                if not os.path.exists(fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\data\{save_file_name}.pkl"):
                    sce.run(cf_params=cf_params, car_params=car_params, has_ui=True)
                sce.plot_scenario(rho=rho)
                sce.indicator_evaluation()
            have_one = True
            break
        if have_one:
            break
