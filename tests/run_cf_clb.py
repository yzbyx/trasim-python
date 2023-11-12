# -*- coding: utf-8 -*-
# @Time : 2023/10/15 11:45
# @Author : yzbyx
# @File : run_cf_clb.py
# Software: PyCharm
import random

import matplotlib.pyplot as plt

from trasim_simplified.core.constant import TrackInfo as TI, Prefix
from trasim_simplified.core.constant import CFM
from trasim_simplified.core.kinematics.cfm import get_cf_func
from trasim_simplified.util.calibrate.clb_cf_model import clb_run, aggregate_result, clb_param_to_df
from trasim_simplified.util.tools import load_from_pickle


if __name__ == '__main__':
    dt = 0.1
    cf_name = CFM.IDM
    cf_func = get_cf_func(cf_name)

    # 生成轨迹，用于验证标定后的参数与原始参数的差异
    # x_lists, v_lists, a_lists, cf_a_lists = get_test_sim_traj(cf_name, cf_func, dt)

    # 读取轨迹
    traj_s_full: dict[str, dict] = load_from_pickle(r"/tests/data/ori_dec_acc_traj_s.pkl")
    traj_s_full = {**traj_s_full["dec"]}
    obs_x_s = [traj_s_full[k][TI.x] for k in traj_s_full.keys()]
    obs_v_s = [traj_s_full[k][TI.v] for k in traj_s_full.keys()]
    obs_lx_s = [traj_s_full[k][Prefix.leader + TI.x] for k in traj_s_full.keys()]
    obs_lv_s = [traj_s_full[k][Prefix.leader + TI.v] for k in traj_s_full.keys()]
    leaderL_s = [traj_s_full[k][Prefix.leader + TI.v_Length].unique()[0] for k in traj_s_full.keys()]
    id_s = [k for k in traj_s_full.keys()]

    # 跟驰模型参数标定
    results = clb_run(cf_func=cf_func, cf_name=cf_name,
                      obs_x_s=obs_x_s, obs_v_s=obs_v_s, obs_lx_s=obs_lx_s, obs_lv_s=obs_lv_s, leaderL_s=leaderL_s,
                      dt=0.1, seed=2023, drawing=0, n_jobs=-1, parallel=True)
    avg_obj, avg_param, std_obj, std_param = aggregate_result(results)
    print(f"avg_obj: {avg_obj}\navg_param: {avg_param}\nstd_obj: {std_obj}\nstd_param: {std_param}")

    df = clb_param_to_df(id_s, results, cf_name)
    df.to_pickle(r"E:\PyProject\process-code-test\tests\data\clb_dec_result.pkl")
