# -*- coding: utf-8 -*-
# @Time : 2023/10/18 15:55
# @Author : yzbyx
# @File : run_control_queue.py
# Software: PyCharm
from trasim_simplified.core.constant import CFM, TrackInfo as TI
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.kinematics.cfm import get_cf_func, get_cf_default_param
from trasim_simplified.util.calibrate.follow_sim import customize_sim, data_to_df

if __name__ == '__main__':
    dt = 0.1
    cf_name = CFM.IDM
    cf_func = get_cf_func(cf_name)
    follow_num = 5

    cf_default_param = get_cf_default_param(cf_name)
    print(f"ori params: {cf_default_param}")

    # 生成轨迹
    x_lists, v_lists, a_lists, cf_a_lists = customize_sim(
        leader_schedule=[(0, 500), (-1, 200), (0, 300), (1, 200), (0, 1000)],
        initial_states=[(25, 20, 0) for i in range(follow_num + 1)],
        length_s=[5] * (follow_num + 1),
        cf_funcs=[cf_func] * follow_num,
        cf_params=[cf_default_param] * follow_num,
        dt=dt, follower_num=follow_num)

    # 绘制轨迹
    df = data_to_df(dt, x_lists, v_lists, a_lists, cf_a_lists)
    Plot.two_dim_plot(df, TI.time, TI.v, marker="x", markersize=1)
    Plot.two_dim_plot(df, TI.time, TI.a, marker="x", markersize=1)
    Plot.two_dim_plot(df, TI.time, TI.cf_acc, marker="x", markersize=1)
    Plot.spatial_time_plot(df, base_line_width=1)
    Plot.show()
