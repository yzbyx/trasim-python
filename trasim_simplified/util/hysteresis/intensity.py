# -*- coding: utf-8 -*-
# @time : 2023/10/22 15:44
# @Author : yzbyx
# @File : intensity.py
# Software: PyCharm
import numpy as np


def cal_project_to_x_axis_area(x_array, y_array):
    """计算轨迹在x轴上的投影面积"""
    assert len(x_array) == len(y_array), f"轨迹xy不一致，x轴点数为{len(x_array)}, y轴点数为{len(y_array)}"
    area = 0
    for i in range(len(x_array) - 1):
        area += (x_array[i + 1] - x_array[i]) * (y_array[i + 1] + y_array[i]) / 2
    return area


def cal_acc_intensity(dec_a, acc_a):
    dec_a = np.array(dec_a)
    acc_a = np.array(acc_a)

    dec_ = np.where(np.abs(dec_a) > 0.2)[0]
    acc_ = np.where(np.abs(acc_a) > 0.2)[0]

    dec_start = dec_[0]
    dec_end = dec_[-1]
    acc_start = acc_[0]
    acc_end = acc_[-1]

    dec_a = dec_a[dec_start: dec_end]
    acc_a = acc_a[acc_start: acc_end]

    return {"dec_a": np.abs(np.mean(dec_a)), "acc_a": np.abs(np.mean(acc_a))}


def cal_sv_intensity(dec_s, dec_v, acc_s, acc_v, cf_e, cf_param: dict):
    """计算净间距-速度平面的迟滞强度， 返回减速迟滞强度，加速迟滞强度，总迟滞强度"""
    dec_s = np.array(dec_s)
    dec_v = np.array(dec_v)
    acc_s = np.array(acc_s)
    acc_v = np.array(acc_v)

    max_speed = np.min([np.max(dec_v), np.max(acc_v)])
    index = np.where(dec_v >= max_speed)[0][-1]
    dec_v = dec_v[index:]  # 找到减速轨迹中最后一个大于等于共同最大速度的点
    dec_s = dec_s[index:]
    index = np.where(acc_v >= max_speed)[0][0]
    acc_v = acc_v[: index]  # 找到加速轨迹中第一个小于等于共同最大速度的点
    acc_s = acc_s[: index]

    # min_speed = np.max([np.min(dec_v), np.min(acc_v)])
    # index = np.where(dec_v <= min_speed)[0][0]
    # dec_v = dec_v[: index]  # 找到减速轨迹中第一个小于等于共同最小速度的点
    # dec_s = dec_s[: index]
    # index = np.where(acc_v <= min_speed)[0][-1]
    # acc_v = acc_v[index:]  # 找到加速轨迹中最后一个小于等于共同最小速度的点
    # acc_s = acc_s[index:]

    # print((np.max(dec_v) - np.min(dec_v)) - (np.max(acc_v) - np.min(acc_v)))
    # assert (np.max(dec_v) - np.min(dec_v)) - (np.max(acc_v) - np.min(acc_v)) < 1, "速度范围不一致"

    # 速度差上的净间距平均
    dec_speed_array = np.linspace(dec_v[-1], dec_v[0], 100)
    e_space_array = np.array([cf_e(**cf_param, speed=speed) for speed in dec_speed_array])
    dec_vs = cal_project_to_x_axis_area(
        np.concatenate([dec_v, dec_speed_array]), np.concatenate([dec_s, e_space_array])
    ) / (np.max(dec_v) - np.min(dec_v))

    acc_speed_array = np.linspace(acc_v[-1], acc_v[0], 100)
    e_space_array = np.array([cf_e(**cf_param, speed=speed) for speed in acc_speed_array])
    acc_vs = cal_project_to_x_axis_area(
        np.concatenate([acc_v, acc_speed_array]), np.concatenate([acc_s, e_space_array])
    ) / (np.max(acc_v) - np.min(acc_v))

    # dec_area = cal_project_to_x_axis_area(dec_v, dec_s)  # -
    # acc_area = cal_project_to_x_axis_area(acc_v, acc_s)  # +
    # e_area = cal_project_to_x_axis_area(speed_array, e_space_array)  # +

    # 时间跨度上的净间距平均
    # e_space_array_dec = np.array([cf_e(**cf_param, speed=speed) for speed in dec_v])
    # dec_delta_s = np.sum(dec_s - e_space_array_dec)
    # a_space_array_acc = np.array([cf_e(**cf_param, speed=speed) for speed in acc_v])
    # acc_delta_s = np.sum(acc_s - a_space_array_acc)

    min_speed = dec_v[-1]

    # "dec_vs", "acc_vs", "total_vs", "dec_ts", "acc_ts", "total_ts", "min_speed", "max_speed", "dec_step", "acc_step"
    return {
        "dec_vs": dec_vs, "acc_vs": acc_vs, "total_vs": dec_vs + acc_vs,
        "dec_t": len(dec_v) * 0.1,
        "acc_t": len(acc_v) * 0.1,
        "dec_avg_acc": (max_speed - min_speed) / (len(dec_v) * 0.1),  # 车辆的平均减速度
        "acc_avg_acc": (max_speed - min_speed) / (len(acc_v) * 0.1),  # 车辆的平均加速度
        "dec_avg_speed": np.mean(dec_v),  # 车辆减速过程的平均速度
        "acc_avg_speed": np.mean(acc_v),  # 车辆加速过程的平均速度
        "min_speed": min_speed,  # 共同最小速度
        "max_speed": max_speed,  # 共同最大速度
        "dv": max_speed - min_speed,  # 速度差
        "dec_init_gap": dec_s[0],  # 减速初始净间距
        "acc_init_gap": acc_s[0],  # 加速初始净间距
    }
