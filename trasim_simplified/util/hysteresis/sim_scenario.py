# -*- coding: utf-8 -*-
# @Time : 2023/11/5 16:56
# @Author : yzbyx
# @File : sim_scenario.py
# Software: PyCharm
import numpy as np

from trasim_simplified.util.calibrate.follow_sim import customize_sim


def slow_to_go_sim(cf_func, cf_param, cf_e=None, init_v=15, dv=10,
                   warmup_time=10, dec_time=None, slow_time=10, acc_time=None, hold_time=10, v_length=5, dt=0.1):
    # print(cf_param, init_v)
    gap = cf_e(**cf_param, speed=init_v) if cf_e is not None else init_v

    warmup_step = round(warmup_time / dt)
    dec_step = round(dec_time / dt) if dec_time is not None else 1
    slow_step = round(slow_time / dt)
    dec_acc = dv / dec_time
    acc_step = round(acc_time / dt) if acc_time is not None else 1
    acc_acc = dv / acc_time
    hold_step = round(hold_time / dt)
    x_lists, v_lists, a_lists, cf_a_lists = customize_sim(
        leader_schedule=[(0, warmup_step), (-dec_acc, dec_step), (0, slow_step), (acc_acc, acc_step), (0, hold_step)],
        initial_states=[(0, init_v, 0), (gap, init_v, 0)],
        length_s=[v_length] * 2,
        cf_funcs=[cf_func],
        cf_params=[cf_param],
        follower_num=1,
        dt=dt
    )
    full_s = np.array(x_lists[0]) - np.array(x_lists[1]) - v_length
    full_v = np.array(v_lists[1])
    full_lv = np.array(v_lists[0])
    dec_s = full_s[warmup_step: warmup_step + dec_step + slow_step]
    dec_v = full_v[warmup_step: warmup_step + dec_step + slow_step]
    acc_s = full_s[warmup_step + dec_step + slow_step: warmup_step + dec_step + slow_step + acc_step + hold_step]
    acc_v = full_v[warmup_step + dec_step + slow_step: warmup_step + dec_step + slow_step + acc_step + hold_step]

    dec_lv = full_lv[warmup_step: warmup_step + dec_step + slow_step]
    acc_lv = full_lv[warmup_step + dec_step + slow_step: warmup_step + dec_step + slow_step + acc_step + hold_step]

    dec_a = a_lists[1][warmup_step: warmup_step + dec_step + slow_step]
    acc_a = a_lists[1][warmup_step + dec_step + slow_step: warmup_step + dec_step + slow_step + acc_step + hold_step]

    dec_lx = np.array(x_lists[0][warmup_step: warmup_step + dec_step + slow_step])
    acc_lx = np.array(x_lists[0][warmup_step + dec_step + slow_step:
                                 warmup_step + dec_step + slow_step + acc_step + hold_step])
    return dec_s, dec_v, acc_s, acc_v, dec_a, acc_a, dec_lv, acc_lv, dec_lx, acc_lx


# def lv_constant_sim(cf_func, cf_param, cf_e=None, init_v=15, dv=10,
#                     warmup_time=10, dec_time=None, slow_time=10, acc_time=None, hold_time=10, v_length=5, dt=0.1):
#     # print(cf_param, init_v)
#     gap = cf_e(**cf_param, speed=init_v) if cf_e is not None else init_v
#
#     warmup_step = round(warmup_time / dt)
#     dec_step = round(dec_time / dt) if dec_time is not None else 1
#     slow_step = round(slow_time / dt)
#     dec_acc = round(dv / dec_time)
#     acc_step = round(acc_time / dt) if acc_time is not None else 1
#     acc_acc = round(dv / acc_time)
#     hold_step = round(hold_time / dt)
#     x_lists, v_lists, a_lists, cf_a_lists = customize_sim(
#         leader_schedule=[(0, warmup_step), (-dec_acc, dec_step), (0, slow_step), (acc_acc, acc_step), (0, hold_step)],
#         initial_states=[(0, init_v, 0), (gap, init_v, 0)],
#         length_s=[v_length] * 2,
#         cf_funcs=[cf_func],
#         cf_params=[cf_param],
#         follower_num=1,
#         dt=dt
#     )
#     full_s = np.array(x_lists[0]) - np.array(x_lists[1]) - v_length
#     full_v = np.array(v_lists[1])
#     dec_s = full_s[warmup_step: warmup_step + dec_step + slow_step]
#     dec_v = full_v[warmup_step: warmup_step + dec_step + slow_step]
#     acc_s = full_s[warmup_step + dec_step + slow_step: warmup_step + dec_step + slow_step + acc_step + hold_step]
#     acc_v = full_v[warmup_step + dec_step + slow_step: warmup_step + dec_step + slow_step + acc_step + hold_step]
#     return dec_s, dec_v, acc_s, acc_v
