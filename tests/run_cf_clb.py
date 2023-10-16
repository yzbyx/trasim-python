# -*- coding: utf-8 -*-
# @Time : 2023/10/15 11:45
# @Author : yzbyx
# @File : run_cf_clb.py
# Software: PyCharm
import random

import matplotlib.pyplot as plt
import pandas as pd

from trasim_simplified.core.constant import TrackInfo as TI
from trasim_simplified.core.constant import CFM, Prefix
from trasim_simplified.core.kinematics.cfm import get_cf_func
from trasim_simplified.util.calibrate.clb_cf_model import clb_run, aggregate_result
from trasim_simplified.util.calibrate.follow_sim import simulation

if __name__ == '__main__':
    cf_name = CFM.IDM
    cf_func = get_cf_func(cf_name)
    traj_s = pd.read_pickle("data/trajectories.pkl")

    # 跟驰模型参数标定
    results = clb_run(cf_func=cf_func, cf_name=cf_name, traj_s=traj_s, dt=0.1,
                      seed=random.randint(0, 9999), drawing=0)
    avg_obj, avg_param, std_obj, std_param = aggregate_result(results)
    print(avg_obj, avg_param, std_obj, std_param)

    # 标定后跟驰模型轨迹仿真
    traj_results = [(i, simulation(cf_func, obs_v=traj[TI.v], obs_x=traj[TI.x],
                                   obs_lx=traj[Prefix.leader + TI.x], obs_lv=traj[Prefix.leader + TI.v], dt=0.1,
                                   cf_params=results[i], leaderL=traj[Prefix.leader + TI.v_Length])) for i, traj in
                    enumerate(traj_s)]

    # 任选一条轨迹进行对比
    traj_test_pos = random.randint(0, len(traj_results) - 1)
    traj_test = traj_results[traj_test_pos]
    traj_step = traj_s[traj_test_pos][TI.time]
    plt.plot(traj_step, traj_s[traj_test_pos][TI.x], label="obs")
    plt.plot(traj_step, traj_test[0], label="sim")
    plt.show()
