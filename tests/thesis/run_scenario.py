# -*- coding: utf-8 -*-
# @Time : 2025/4/16 15:33
# @Author : yzbyx
# @File : run_scenario.py
# Software: PyCharm
from matplotlib import pyplot as plt

from trasim_simplified.core.constant import ScenarioTraj, ScenarioMode
from trasim_simplified.util.scenario.scenario_loader import Scenario
from trasim_simplified.util.tools import load_from_pickle

if __name__ == '__main__':
    scenario_path = r"E:\BaiduSyncdisk\weaving-analysis\data\pattern_scenario_data.pkl"
    scenario_data: dict[str, list[ScenarioTraj]] = load_from_pickle(scenario_path)  # pattern_name, pattern_traj_s

    for pattern_name, pattern_traj_s in scenario_data.items():
        for pattern_traj in pattern_traj_s:
            # if pattern_traj.dataset_name != "NGSIM":
            #     continue
            # if pattern_traj.track_id not in [24571]:
            #     continue
            try:
                print(f"Pattern Name: {pattern_name}, Trajectory ID: {pattern_traj.track_id}")
                sce = Scenario(pattern_traj)
                sce.run(mode=ScenarioMode.NO_INTERACTION)
            except Exception as e:
                print(f"Error: {e}")
                continue