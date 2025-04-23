# -*- coding: utf-8 -*-
# @time : 2024/6/11 20:54
# @Author : yzbyx
# @File : util.py
# Software: PyCharm
from typing import TYPE_CHECKING

import numpy as np


def get_x_guess(T, state):
    """根据猜测的终点x方向状态确定纵向五次多项式fxt参数"""
    A = np.array([
        [1, 0, 0, 0, 0, 0],  # x0
        [0, 1, 0, 0, 0, 0],  # dx0
        [0, 0, 2, 0, 0, 0],  # ddx0
        [1, T, T ** 2, T ** 3, T ** 4, T ** 5],  # x1
        [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],  # dx1
        [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]  # ddx1
    ])
    xf = state[0] + state[1] * T
    dxf = state[1]
    ddxf = 0
    b = np.concatenate([state[:3], [xf, dxf, ddxf]])
    x = np.linalg.solve(A, b)
    return x, xf, dxf, ddxf


def get_y_guess(T, state, yf):
    """根据猜测的终点y方向状态确定纵向五次多项式fyt参数"""
    A = np.array([
        [1, 0, 0, 0, 0, 0],  # y0
        [0, 1, 0, 0, 0, 0],  # dy0
        [0, 0, 2, 0, 0, 0],  # ddy0
        [1, T, T ** 2, T ** 3, T ** 4, T ** 5],  # y1
        [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],  # dy1
        [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]  # ddy1
    ])
    dyf = 0
    ddyf = 0
    b = np.concatenate([state[3:6], [yf, dyf, ddyf]])
    y = np.linalg.solve(A, b)
    return y, yf, dyf, ddyf


def get_xy_quintic(x, t):
    """返回[x, vx, ax, y, vy, ay]"""
    return np.array([
        [1, t, t ** 2, t ** 3, t ** 4, t ** 5, 0, 0, 0, 0, 0, 0],
        [0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, t, t ** 2, t ** 3, t ** 4, t ** 5],
        [0, 0, 0, 0, 0, 0, 0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3]
    ]) @ x


def get_linear(order, max_speed, dec_max):
    """"刹停距离函数的线性化，dist = av + b，返回a，b"""
    ab_s = []
    stamp = np.linspace(0, max_speed, order + 1)
    for i in range(order):
        v0, v1 = stamp[i], stamp[i + 1]
        d0 = v0 ** 2 / (2 * abs(dec_max))
        d1 = v1 ** 2 / (2 * abs(dec_max))
        a = (d1 - d0) / (v1 - v0)
        b = d0 - a * v0
        ab_s.append([a, b])
    return ab_s


def get_v_square_linear(v, l_v, max_v):
    """
     v * (v - dxt_PC[:step])线性化
    :param v: 当前速度
    :param l_v: 线性化点
    :return: 线性化后的速度平方
    """
    point_1 = (0, 0)
    point_2 = (l_v / 2, )
    f_1 = lambda v_, l_v_: - 1 / 2 * l_v_ * v_
    f_2 = lambda v_, l_v_: (max_v ** 2 - max_v * l_v_ + 1 / 4 * l_v_ ** 2) / (max_v - l_v_ / 2) * v_



def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps


def correct_steering(angle):
    """转为[-np.pi / 2, np.pi / 2]"""
    angle[angle > np.pi / 2] -= np.pi
    return angle


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    copied from https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/stanley_control/stanley_control.html
    """
    # while angle > np.pi:
    #     angle -= 2.0 * np.pi
    #
    # while angle < -np.pi:
    #     angle += 2.0 * np.pi

    angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi

    return angle


def get_state_space(dt, ref_yaw, ref_delta, ref_v, L):
    """
    状态空间方程，以后轴中心
    """
    # 以后轴中心
    # A = np.array([
    #     [1.0, 0.0, -ref_v * dt * np.sin(ref_yaw)],
    #     [0.0, 1.0, ref_v * dt * np.cos(ref_yaw)],
    #     [0.0, 0.0, 1.0]]
    # )
    # B = np.array([
    #     [dt * np.cos(ref_yaw), 0],
    #     [dt * np.sin(ref_yaw), 0],
    #     [dt * np.tan(ref_delta) / L, ref_v * dt / (L * np.cos(ref_delta) * np.cos(ref_delta))]]
    # )

    # 以质心为中心
    ref_beta = np.arctan(0.5 * np.tan(ref_delta))
    A = np.array([
        [1.0, 0.0, -ref_v * dt * np.sin(ref_yaw + ref_beta)],
        [0.0, 1.0, ref_v * dt * np.cos(ref_yaw + ref_beta)],
        [0.0, 0.0, 1.0]
    ])
    c = 1 / (1 + (0.5 * np.tan(ref_delta)) ** 2) * (1 / (2 * np.cos(ref_delta) ** 2))
    B = np.array([
        [dt * np.cos(ref_yaw + ref_beta), -ref_v * np.sin(ref_yaw + ref_beta) * dt * c],
        [dt * np.sin(ref_yaw + ref_beta), ref_v * np.cos(ref_yaw + ref_beta) * dt * c],
        [dt * np.sin(ref_beta) / L, ref_v * dt * np.cos(ref_beta) * c / L]
    ])
    return A, B


def interval_intersection(interval1, interval2, print_flag=False):
    a1, b1 = interval1
    a2, b2 = interval2

    # 计算交集的开始和结束
    start = round(np.maximum(a1, a2), 3)
    end = round(np.minimum(b1, b2), 3)

    # 判断是否有交集
    if start <= end:
        return start, end
    else:
        if print_flag:
            print(f"intersection none: {interval1}, {interval2}")
            raise ValueError("没有交集")
        return None  # 没有交集
