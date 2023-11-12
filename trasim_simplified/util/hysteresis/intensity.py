# -*- coding: utf-8 -*-
# @Time : 2023/10/22 15:44
# @Author : yzbyx
# @File : intensity.py
# Software: PyCharm
import numpy as np

from trasim_simplified.core.kinematics.cfm import get_cf_equilibrium


def cal_project_to_x_axis_area(x_array, y_array):
    """计算轨迹在x轴上的投影面积"""
    assert len(x_array) == len(y_array), f"轨迹xy不一致，x轴点数为{len(x_array)}, y轴点数为{len(y_array)}"
    area = 0
    for i in range(len(x_array) - 1):
        area += (x_array[i + 1] - x_array[i]) * y_array[i]
    return area


def cal_sv_intensity(dec_s, dec_v, acc_s, acc_v, cf_e, cf_param: dict):
    """计算净间距-速度平面的迟滞强度， 返回减速迟滞强度，加速迟滞强度，总迟滞强度"""
    dec_s = np.array(dec_s)
    dec_v = np.array(dec_v)
    acc_s = np.array(acc_s)
    acc_v = np.array(acc_v)

    min_speed = dec_v.min()  # 由于轨迹连续，因此只需dec或acc一侧轨迹即可
    max_speed = np.min([dec_v[0], acc_v[-1]])
    index = np.where(dec_v >= max_speed)[0][-1]
    dec_v = dec_v[index:]  # 找到减速轨迹中最后一个大于等于共同最大速度的点
    dec_s = dec_s[index:]
    index = np.where(acc_v >= max_speed)[0][0]
    acc_v = acc_v[: index]  # 找到加速轨迹中第一个小于等于共同最大速度的点
    acc_s = acc_s[: index]

    # 速度差上的净间距平均
    speed_array = np.linspace(min_speed, max_speed, 100)
    e_space_array = np.array([cf_e(**cf_param, speed=speed) for speed in speed_array])

    dec_area = cal_project_to_x_axis_area(dec_v, dec_s)  # -
    acc_area = cal_project_to_x_axis_area(acc_v, acc_s)  # +
    e_area = cal_project_to_x_axis_area(speed_array, e_space_array)  # +

    # 时间跨度上的净间距平均
    e_space_array_dec = np.array([cf_e(**cf_param, speed=speed) for speed in dec_v])
    dec_delta_s = np.sum(dec_s - e_space_array_dec)
    a_space_array_acc = np.array([cf_e(**cf_param, speed=speed) for speed in acc_v])
    acc_delta_s = np.sum(acc_s - a_space_array_acc)

    speed_range = max_speed - min_speed
    time_step_range = len(dec_v) + len(acc_v)

    # "dec_vs", "acc_vs", "total_vs", "dec_ts", "acc_ts", "total_ts", "min_speed", "max_speed", "dec_step", "acc_step"
    return {"dec_vs": (dec_area + e_area) / speed_range, "acc_vs": (acc_area - e_area) / speed_range,
            "total_vs": (acc_area + dec_area) / speed_range,
            "dec_ts": dec_delta_s / len(dec_s), "acc_ts": acc_delta_s / len(acc_s),
            "total_ts": (acc_delta_s - dec_delta_s) / time_step_range,
            "dec_avg_acc": (max_speed - min_speed) / (len(dec_v) * 0.1),
            "acc_avg_acc": (max_speed - min_speed) / (len(acc_v) * 0.1),
            "dec_avg_speed": np.mean(dec_v), "acc_avg_speed": np.mean(acc_v),
            "min_speed": min_speed, "max_speed": max_speed, "dv": max_speed - min_speed,
            "dec_step": len(dec_v), "acc_step": len(acc_v)}
