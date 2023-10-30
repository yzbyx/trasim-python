# -*- coding: utf-8 -*-
# @Time : 2023/10/14 21:25
# @Author : yzbyx
# @File : follow_sim.py
# Software: PyCharm
import typing

import numpy as np
import pandas as pd

from trasim_simplified.core.constant import TrackInfo as TI

from trasim_simplified.msg.trasimError import TrasimError


def simulation(cf_func, init_x, init_v, obs_lx, obs_lv, cf_param, dt,
               leaderL: float | typing.Iterable, update_method="Euler") -> tuple[list, list, list, list]:
    """
    给定前车的位置、速度，以及模型参数，仿真得到后车的位置、速度、加速度，默认前车ID不变

    注意！原始数据的位置、速度的关系需要与模型状态的更新方式相对应
    """
    tau = cf_param.get('tau', False)
    if tau:
        assert tau >= dt
    speed = float(init_v)
    pos = float(init_x)
    sim_pos, sim_speed, sim_acc, sim_cf_acc = [pos], [speed], [0], [0]
    if isinstance(leaderL, float | int | np.float64 | np.float32):
        leaderL = [leaderL] * len(obs_lx)
    for lx, lv, ll in zip(obs_lx[:-1], obs_lv[:-1], leaderL[:-1]):
        cf_acc = cf_func(**cf_param, speed=speed, gap=lx - pos - ll, leaderV=lv, interval=dt)
        sim_cf_acc.append(cf_acc)
        if update_method == "Euler":  # 差分型，最常用
            speed_before = speed
            speed += cf_acc * dt
            if speed < 0:
                speed = 0
            cf_acc = (speed - speed_before) / dt
            pos += speed * dt
        elif update_method == "Ballistic":  # Treiber et al., 2006; Treiber and Kanagaraj, 2015 速度渐变型
            speed_before = speed
            speed += cf_acc * dt
            if speed < 0:
                speed = 0
            cf_acc = (speed - speed_before) / dt
            pos += (speed + speed_before) * dt / 2
        else:
            raise TrasimError("update_method must be 'Euler' or 'Ballistic'")
        sim_pos.append(pos)
        sim_speed.append(speed)
        sim_acc.append(cf_acc)
    return sim_pos, sim_speed, sim_acc, sim_cf_acc


def customize_sim(leader_schedule: list[tuple[float, int]], initial_states: list[tuple[float, float, float]],
                  length_s: list[float], cf_funcs: list[typing.Callable], cf_params: list[dict],
                  follower_num=1, dt=0.1):
    """
    自定义头车加速度的后车轨迹仿真

    :param leader_schedule: 头车加速度的时间序列 [(a, step), ...] a:加速度，step:持续时间步
    :param initial_states: 后车初始状态 [(gap, v, a), ...] gap:净间距，v:速度，a:加速度
    :param length_s: 车辆长度
    :param cf_funcs: 跟驰模型
    :param cf_params: 跟驰模型参数
    :param follower_num: 后车数量
    :param dt: 仿真步长
    """
    assert follower_num > 0, "follower_num must be greater than 0"
    assert len(length_s) == follower_num + 1, "length_s must be the same length as follower_num + 1"
    assert len(initial_states) == follower_num + 1, "initial_state must be the same length as follower_num + 1"
    assert len(leader_schedule) > 0, "leader_schedule must be not empty"
    assert len(cf_funcs) == follower_num, "cf_func must be the same length as follower_num"
    assert len(cf_params) == follower_num, "cf_params must be the same length as follower_num"

    x_list, v_list, a_list, cf_a_list = generate_traj(leader_schedule, initial_states[0], dt)
    current_x = x_list[0]
    x_lists, v_lists, a_lists, cf_a_lists = [x_list], [v_list], [a_list], [cf_a_list]
    for (gap, v, a), leader_length, cf_func, cf_param in zip(initial_states[1:], length_s[:-1], cf_funcs, cf_params):
        current_x -= (gap + leader_length)
        sim_pos, sim_speed, sim_acc, sim_cf_acc = simulation(cf_func, init_x=current_x, init_v=v, obs_lv=v_lists[-1],
                                                             obs_lx=x_lists[-1], cf_param=cf_param, dt=dt,
                                                             leaderL=leader_length)
        x_lists.append(sim_pos)
        v_lists.append(sim_speed)
        a_lists.append(sim_acc)
        cf_a_lists.append(sim_cf_acc)
    return x_lists, v_lists, a_lists, cf_a_lists


def generate_traj(schedule: list[tuple[float, float]], initial_state: tuple[float, float, float], dt=0.1):
    x, v, a = initial_state
    x_list = [x]
    v_list = [v]
    a_list = [a]
    cf_a_list = [0]
    for a, step in schedule:
        for _ in range(int(step)):
            v += a * dt
            real_a = a
            if v < 0:
                v = 0
                real_a = (v - v_list[-1]) / dt
            x += v * dt
            x_list.append(x)
            v_list.append(v)
            a_list.append(real_a)
            cf_a_list.append(a)
    return x_list, v_list, a_list, cf_a_list


def data_to_df(dt: float, x_lists, v_lists, a_lists, cf_a_lists):
    if not isinstance(x_lists[0], typing.Iterable):
        x_lists = [x_lists]
        v_lists = [v_lists]
        a_lists = [a_lists]
        cf_a_lists = [cf_a_lists]
    step_total = len(x_lists[0])
    car_total = len(x_lists)
    df = pd.DataFrame(
        data={TI.v_ID: np.concatenate([[i] * step_total for i in range(car_total)]),
              TI.Frame_ID: list(range(step_total)) * car_total,
              TI.x: np.concatenate(x_lists),
              TI.v: np.concatenate(v_lists),
              TI.a: np.concatenate(a_lists),
              TI.cf_acc: np.concatenate(cf_a_lists)}
    )
    df[TI.time] = df[TI.Frame_ID] * dt
    return df
