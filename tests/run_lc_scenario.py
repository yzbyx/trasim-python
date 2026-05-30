# @Time : 2025/9/6 9:22
# @Author : yzbyx
# @File : run_lc_scenario.py
# Software: PyCharm
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from trasim_simplified.util.scenario.scenario_interaction import LcScenario
from trasim_simplified.util.scenario.scenario_util import relation_2_id_car_params

np.random.seed(2025)

from trasim_simplified.core.agent.game_agent import Game_Vehicle, Game_A_Vehicle
from trasim_simplified.core.constant import ScenarioMode, COLOR, ScenarioTraj, V_CLASS, RouteType, SurrClass, CFM
from trasim_simplified.util.tools import load_from_pickle, save_to_pickle

if __name__ == '__main__':
    """
    换道场景模拟
    
    设定周边车辆类型（surr_class）、HV激进度、AV合作系数进行仿真
    """
    cwd = os.getcwd()
    scenario_path = os.path.join(cwd, "data", "pattern_scenario_data.pkl")
    scenario_data: dict[str, list[ScenarioTraj]] = load_from_pickle(scenario_path)  # pattern_name, pattern_traj_s

    for pattern_name, pattern_traj_s in scenario_data.items():
        only_run_once = False
        for pattern_traj in pattern_traj_s:
            name = f"{pattern_traj.dataset_name}_{pattern_name}_{pattern_traj.track_id}"
            # 想要跳过的场景
            if name in [
                "CitySim_驶出_2717", "CitySim_驶出_14385", "NGSIM_驶入_1257",
                "CitySim_驶入_959", "CitySim_驶入_5053", "CitySim_驶入_2014",
                "CitySim_驶入_4782", "CitySim_驶入_955", "CitySim_驶入_2130",
                "CitySim_驶入_20307", "NGSIM_预驶出_1547", "CitySim_预驶出_2343",
                "CitySim_预驶出_2141", "CitySim_松弛行为_3122", "CitySim_松弛行为_21140",
                "CitySim_预期行为_778"
            ]:
                continue

            # 跳过存在周边车辆缺失的场景
            if pattern_traj.CP_traj is None:
                continue
            if pattern_traj.TR_traj is None or pattern_traj.TP_traj is None:
                continue
            if len(pattern_traj.TR_traj) > 70 and pattern_traj.dataset_name == "NGSIM":
                continue
            if len(pattern_traj.TP_traj) > 70 * 3 and pattern_traj.dataset_name == "CitySim":
                continue

            # 跳过跟车距离过近的场景
            if (pattern_traj.EV_traj["myLocalLon"].values[0] -
                    pattern_traj.TR_traj["myLocalLon"].values[0] > 20):
                continue
            if (pattern_traj.EV_traj["myLocalLon"].values[0] -
                    pattern_traj.TP_traj["myLocalLon"].values[0] < -20):
                continue

            print(name, "tr_id", pattern_traj.TR_traj["trackId"].values[0],
                  "tp_id", pattern_traj.TP_traj["trackId"].values[0])
            base_path = os.path.join(cwd, "data")

            # 配置目标车道前后车类型
            tr_type = V_CLASS.GAME_HV
            tp_type = V_CLASS.GAME_HV
            tr_co = tp_co = 0.5  # 前车与后车的合作系数，仅当tr、tp为AV时起作用
            osm_path = os.path.join(cwd, "data", "US-101", f"map.osm")

            surr_class = SurrClass(
                ev_type=V_CLASS.GAME_AV,
                tr_type=V_CLASS.GAME_HV,
                tp_type=V_CLASS.GAME_HV,
            )

            sce = LcScenario(
                pattern_traj,
                osm_path,
                av_cf=CFM.TPACC,
                hv_cf=CFM.KK
            )

            # 标定每辆车的跟驰参数
            cf_params = {}
            cf_params_path = os.path.join(base_path, f"{name}_cf_params_TR-{tr_type}_TP-{tp_type}.pkl")
            if not os.path.exists(cf_params_path):
                cf_params = sce.calibrate_cf(surr_class)
                save_to_pickle(
                    cf_params,
                    cf_params_path
                )
            cf_params = load_from_pickle(cf_params_path)
            print(cf_params)

            # 标定博弈相关参数
            car_params = {}
            car_params_path = os.path.join(base_path, f"{name}_car_params.pkl")
            if not os.path.exists(car_params_path):
                car_params = sce.opti_ade(surr_class, cf_params)
                save_to_pickle(
                    car_params,
                    car_params_path
                )
            car_params = load_from_pickle(car_params_path)
            car_params_addition = relation_2_id_car_params(
                sce,
                {"TR": {"game_co": tr_co}, "TP": {"game_co": tp_co}}
            )
            car_params.update(car_params_addition)
            print(car_params)

            save_file_name = (
                f"{name}"
                f"_TR-{surr_class.tr_type}-{tr_co}"
                f"_TP-{surr_class.tp_type}-{tp_co}"
            )
            print(save_file_name)
            save_file_path = os.path.join(base_path, f"{save_file_name}.pkl")
            if not os.path.exists(save_file_path):
                res_df = sce.run(
                    surr_class,
                    cf_params=cf_params,
                    car_params=car_params,
                    has_ui=True,
                    save_res=True,
                    save_file_path=save_file_path
                )

            sce.plot_scenario(res_df_path=save_file_path, highlight=["EV"])
            sce.indicator_evaluation(surr_class, save_file_path=save_file_path)

            only_run_once = True
            break
        if only_run_once:
            break
