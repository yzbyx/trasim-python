# -*- coding: utf-8 -*-
# @Time : 2025/4/16 15:33
# @Author : yzbyx
# @File : run_scenario_real.py
# Software: PyCharm
import os

from trasim_simplified.core.constant import ScenarioTraj, ScenarioMode, V_CLASS, SurrClass
from trasim_simplified.util.scenario.scenario_loader import Scenario
from trasim_simplified.util.tools import load_from_pickle, save_to_pickle

if __name__ == '__main__':
    scenario_path = r"E:\BaiduSyncdisk\weaving-analysis\data\pattern_scenario_data.pkl"
    scenario_data: dict[str, list[ScenarioTraj]] = load_from_pickle(scenario_path)  # pattern_name, pattern_traj_s

    for pattern_name, pattern_traj_s in scenario_data.items():
        have_one = False
        for pattern_traj in pattern_traj_s:
            name = f"{pattern_traj.dataset_name}_{pattern_name}_{pattern_traj.track_id}"
            if name not in ["CitySim_交织换道_208"]:
                continue
            base_path = r"E:\BaiduSyncdisk\car-following-model\tests\thesis\data"
            surr_class = SurrClass(
                ev_type=V_CLASS.GAME_AV, tr_type=V_CLASS.GAME_HV
            )
            sce = Scenario(pattern_traj, surr_class)
            name = name + "_" + surr_class.name()
            print(name)
            sce.save_file_name = name

            if not os.path.exists(fr"{base_path}\{name}_cf_params.pkl"):
                cf_params = sce.calibrate_cf()
                save_to_pickle(cf_params, fr"{base_path}\{name}_cf_params.pkl")
            cf_params = load_from_pickle(fr"{base_path}\{name}_cf_params.pkl")

            car_params = {}
            if not os.path.exists(fr"{base_path}\{name}_car_params.pkl"):
                car_params = sce.opti_ade(cf_params)
                save_to_pickle(car_params,fr"{base_path}\{name}_car_params.pkl")
            car_params = load_from_pickle(fr"{base_path}\{name}_car_params.pkl")

            if not os.path.exists(fr"E:\BaiduSyncdisk\car-following-model\tests\thesis\data\{name}_sim_data.pkl"):
                sce.run(cf_params=cf_params, car_params=car_params, has_ui=True)
            sce.plot_scenario()
            sce.indicator_evaluation()

            have_one = True
            break
        if have_one:
            break
