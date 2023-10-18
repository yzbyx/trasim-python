# -*- coding: utf-8 -*-
# @Time : 2023/10/15 11:45
# @Author : yzbyx
# @File : run_cf_clb.py
# Software: PyCharm
import random

import matplotlib.pyplot as plt

from trasim_simplified.core.constant import TrackInfo as TI
from trasim_simplified.core.constant import CFM
from trasim_simplified.core.kinematics.cfm import get_cf_func, get_cf_default_param
from trasim_simplified.util.calibrate.clb_cf_model import clb_run, aggregate_result, cf_param_ranges
from trasim_simplified.util.calibrate.follow_sim import simulation, customize_sim, data_to_df
from trasim_simplified.util.calibrate.gof_func import RMSE

if __name__ == '__main__':
    dt = 0.1
    cf_name = CFM.IDM
    cf_func = get_cf_func(cf_name)

    param_ord = list(cf_param_ranges[cf_name].keys())
    cf_default_param = {name: get_cf_default_param(cf_name)[name] for name in param_ord}
    print(f"ori params: {cf_default_param}")

    # 生成轨迹
    x_lists, v_lists, a_lists = customize_sim(leader_schedule=[(0, 300), (-1, 200), (0, 100), (1, 200), (0, 300)],
                                              initial_states=[(0, 20, 0), (10, 20, 0)],
                                              length_s=[5, 5], cf_funcs=[cf_func], cf_params=[cf_default_param], dt=dt)
    traj_s = [{"obs_x": x_lists[1], "obs_v": v_lists[1], "obs_lx": x_lists[0], "obs_lv": v_lists[0], "leaderL": 5}]
    rmse = RMSE(sim_x=x_lists[1], sim_v=v_lists[1],
                obs_x=x_lists[1], obs_v=v_lists[1], obs_lx=5, eval_params=['dhw'])
    print(f"ori RMSE: {rmse}")

    # 读取轨迹
    # traj_s = pd.read_pickle("data/trajectories.pkl")

    # 跟驰模型参数标定
    results = clb_run(cf_func=cf_func, cf_name=cf_name, traj_s=traj_s,
                      dt=0.1, seed=random.randint(0, 9999), drawing=0)
    avg_obj, avg_param, std_obj, std_param = aggregate_result(results)
    print(f"avg_obj: {avg_obj}\navg_param: {avg_param}\nstd_obj: {std_obj}\nstd_param: {std_param}")

    # 标定后跟驰模型轨迹仿真
    traj_results = [(i, data_to_df(*simulation(cf_func, init_v=traj["obs_v"][0], init_x=traj["obs_x"][0],
                                   obs_lx=traj["obs_lx"], obs_lv=traj["obs_lv"], dt=0.1,
                                   cf_param=results[i]["Vars"],
                                   leaderL=traj["leaderL"]), dt)) for i, traj in enumerate(traj_s)]

    # 任选一条轨迹进行对比
    traj_test_pos = random.randint(0, len(traj_results) - 1)
    _, traj_test = traj_results[traj_test_pos]
    traj_step = traj_test[TI.time]
    plt.plot(traj_step, traj_s[traj_test_pos]["obs_x"], label="obs")
    plt.plot(traj_step, traj_test[TI.x], label="sim")
    plt.legend()
    plt.show()
