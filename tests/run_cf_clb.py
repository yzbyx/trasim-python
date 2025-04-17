# -*- coding: utf-8 -*-
# @time : 2023/10/15 11:45
# @Author : yzbyx
# @File : run_cf_clb.py
# Software: PyCharm
import random

import matplotlib.pyplot as plt
import pandas as pd

from trasim_simplified.core.constant import TrackInfo as TI, Prefix
from trasim_simplified.core.constant import CFM
from trasim_simplified.core.kinematics.cfm import get_cf_func, get_cf_model
from trasim_simplified.util.calibrate.clb_cf_model import clb_run, aggregate_result, clb_param_to_df
from trasim_simplified.util.tools import load_from_pickle


if __name__ == '__main__':
    dt = 0.1
    cf_name = CFM.KK
    cf_func = get_cf_model(CFM.KK)

    # 生成轨迹，用于验证标定后的参数与原始参数的差异
    # x_lists, v_lists, a_lists, cf_a_lists = get_test_sim_traj(cf_name, cf_func, dt)

    # 读取轨迹
    traj_s_full = pd.read_pickle(r"E:\BaiduSyncdisk\data\NGSIM\NGSIM_follow_pair.pkl")
    # traj_s_full = {**traj_s_full["dec"]}
    # obs_x_s = [traj_s_full[k][TI.x] for k in traj_s_full.keys()]
    # obs_v_s = [traj_s_full[k][TI.v] for k in traj_s_full.keys()]
    # obs_lx_s = [traj_s_full[k][Prefix.leader + TI.x] for k in traj_s_full.keys()]
    # obs_lv_s = [traj_s_full[k][Prefix.leader + TI.v] for k in traj_s_full.keys()]
    # leaderL_s = [traj_s_full[k][Prefix.leader + TI.length].unique()[0] for k in traj_s_full.keys()]
    # id_s = [k for k in traj_s_full.keys()]
    obs_x_s = []
    obs_v_s = []
    obs_a_s = []
    obs_lx_s = []
    obs_lv_s = []
    obs_la_s = []
    leaderL_s = []
    id_s = []
    num = 0
    for pair_id in traj_s_full["pairId"].unique():
        obs_x_s.append(traj_s_full["myLocalLon"])
        obs_v_s.append(traj_s_full["myLonVelocity"])
        obs_a_s.append(traj_s_full["myLonAcceleration"])
        obs_lx_s.append(traj_s_full["myLeadLocalLon"])
        obs_lv_s.append(traj_s_full["myLeadLonVelocity"])
        obs_la_s.append(traj_s_full["myLeadLonAcceleration"])
        leaderL_s.append(traj_s_full["myLeadLength"].unique()[0])
        id_s.append(pair_id)
        num += 1
        if num > 100:
            break

    # 跟驰模型参数标定
    results = clb_run(cf_func=cf_func, cf_name=cf_name,
                      obs_x_s=obs_x_s, obs_v_s=obs_v_s, obs_a_s=obs_a_s,
                      obs_lx_s=obs_lx_s, obs_lv_s=obs_lv_s, obs_la_s=obs_la_s,
                      leaderL_s=leaderL_s,
                      dt=0.1, seed=2023, drawing=0, n_jobs=-1, parallel=True)
    avg_obj, avg_param, std_obj, std_param = aggregate_result(results)
    print(f"avg_obj: {avg_obj}\navg_param: {avg_param}\nstd_obj: {std_obj}\nstd_param: {std_param}")

    df = clb_param_to_df(id_s, results, cf_name)
    df.to_pickle(r"E:\PyProject\process-code-test\tests\data\clb_dec_result.pkl")
