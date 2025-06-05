# -*- coding: utf-8 -*-
# @time : 2024/12/18 16:30
# @Author : yzbyx
# @File : ref_path.py
# Software: PyCharm
import numpy as np

from trasim_simplified.core.agent.vehicle import Vehicle
from trasim_simplified.core.constant import TrajPoint

np.seterr(divide='ignore', invalid='ignore')


def get_backward_diff(x_pos, y_pos, dt):
    xv = np.diff(x_pos) / dt
    xv = np.insert(xv, 0, xv[0])

    yv = np.diff(y_pos) / dt
    yv = np.insert(yv, 0, yv[0])

    xa = np.diff(xv) / dt
    xa = np.insert(xa, 0, 0)
    ya = np.diff(yv) / dt
    ya = np.insert(ya, 0, 0)

    return xv, yv, xa, ya


def get_center_diff(x_pos, y_pos, dt, frame=None):
    if frame is None:
        xv = [(x_pos[i + 1] - x_pos[i - 1]) / (2 * dt) for i in range(1, len(x_pos) - 1)]
        yv = [(y_pos[i + 1] - y_pos[i - 1]) / (2 * dt) for i in range(1, len(y_pos) - 1)]
        xv = np.insert(xv, 0, xv[0])
        yv = np.insert(yv, 0, yv[0])
        xv = np.insert(xv, len(xv), xv[-1])
        yv = np.insert(yv, len(yv), yv[-1])

        xa = [(x_pos[i + 1] - 2 * x_pos[i] + x_pos[i - 1]) / (dt ** 2) for i in range(1, len(x_pos) - 1)]
        ya = [(y_pos[i + 1] - 2 * y_pos[i] + y_pos[i - 1]) / (dt ** 2) for i in range(1, len(y_pos) - 1)]
        xa = np.insert(xa, 0, 0)
        ya = np.insert(ya, 0, 0)
        xa = np.insert(xa, len(xa), 0)
        ya = np.insert(ya, len(ya), 0)
    else:
        t = [(frame[i] - frame[i - 1]) * dt for i in range(1, len(frame))]
        xv = [(x_pos[i + 1] - x_pos[i - 1]) / (t[i] + t[i - 1]) for i in range(1, len(x_pos) - 1)]
        yv = [(y_pos[i + 1] - y_pos[i - 1]) / (t[i] + t[i - 1]) for i in range(1, len(y_pos) - 1)]
        xv = np.insert(xv, 0, xv[0])
        yv = np.insert(yv, 0, yv[0])
        xv = np.insert(xv, len(xv), xv[-1])
        yv = np.insert(yv, len(yv), yv[-1])

        xa = [(xv[i + 1] - xv[i - 1]) / (t[i] + t[i - 1]) for i in range(1, len(xv) - 1)]
        ya = [(yv[i + 1] - yv[i - 1]) / (t[i] + t[i - 1]) for i in range(1, len(yv) - 1)]
        xa = np.insert(xa, 0, 0)
        ya = np.insert(ya, 0, 0)
        xa = np.insert(xa, len(xa), 0)
        ya = np.insert(ya, len(ya), 0)

    return xv, yv, xa, ya


class ReferencePath:
    def __init__(self, ref_path, dt):
        self.ref_x = ref_path[:, 0]
        self.ref_dx = ref_path[:, 1]
        self.ref_ddx = ref_path[:, 2]
        self.ref_y = ref_path[:, 3]
        self.ref_dy = ref_path[:, 4]
        self.ref_ddy = ref_path[:, 5]
        self.ref_path = ref_path

        self.dt = dt

        self.ref_v = None
        self.ref_yaw = None
        self.ref_k = None
        self.ref_a = None
        self.ref_delta = None

        self.ref_len = None
        """相邻参考点的距离"""

        self.length = len(self.ref_x)

    def cal_ref(self):
        """
        :return: ref_v, ref_yaw, ref_k, ref_delta
        """
        # xv_b, yv_b, xa_b, ya_b = (
        #     get_backward_diff(self.ref_x, self.ref_y, self.dt))
        xv_b = np.diff(self.ref_x) / self.dt
        xv_b = np.insert(xv_b, 0, self.ref_dx[0])

        yv_b = np.diff(self.ref_y) / self.dt
        yv_b = np.insert(yv_b, 0, self.ref_dy[0])

        xa_b = np.diff(xv_b) / self.dt
        xa_b = np.insert(xa_b, 0, self.ref_ddx[0])
        ya_b = np.diff(yv_b) / self.dt
        ya_b = np.insert(ya_b, 0, self.ref_ddy[0])

        x_pos = self.ref_x
        y_pos = self.ref_y
        dt = self.dt
        xv_c, yv_c, xa_c, ya_c = get_center_diff(x_pos, y_pos, dt)

        # 速度和加速度为后向差分
        self.ref_v = np.linalg.norm(np.vstack((xv_b, yv_b)), axis=0)  # 参考速度
        ref_a = np.diff(self.ref_v) / self.dt  # 参考加速度
        self.ref_a = np.insert(ref_a, 0, 0)  # 插入第一个点的加速度(不影响结果)

        self.ref_yaw = np.arctan2(yv_c, xv_c)  # 参考航向角

        # kappa = r'(t) x r''(t) / |r'(t)|^3
        self.ref_k = ((ya_b * xv_b - xa_b * yv_b) / (
                (xv_b ** 2 + yv_b ** 2) ** (3 / 2))
             )  # 曲率k计算
        # self.ref_k = ((ya_c * xv_c - xa_c * yv_c) /
        #               ((xv_c ** 2 + yv_c ** 2) ** (3 / 2)))  # 曲率k计算

    def cal_ref_delta(self, ugv: Vehicle):
        k = self.ref_k
        min_rear_r = ugv.wheelbase / np.tan(ugv.DELTA_MAX)
        min_head_r = np.sqrt((ugv.prop_ * ugv.wheelbase) ** 2 + min_rear_r ** 2)
        threshold = 1 / min_head_r
        k[(k < - threshold)] = - threshold  # 曲率限制，避免超出车辆物理限制
        k[(k > threshold)] = threshold  # 曲率限制，避免超出车辆物理限制
        # delta_ref = np.arctan(1 / ugv.prop_ * np.tan(np.arcsin(k * ugv.wheel_base * ugv.prop_)))
        rear_r = np.sqrt((1 / k) ** 2 - (ugv.wheelbase * ugv.prop_) ** 2)
        sign_k = np.sign(k)
        delta_ref = np.arctan(ugv.wheelbase / rear_r * sign_k)

        self.ref_delta = delta_ref

        self.check_ref()

    def check_ref(self):
        assert np.all(~np.isnan(self.ref_v)), self.ref_v
        assert np.all(~np.isnan(self.ref_a)), self.ref_a
        assert np.all(~np.isnan(self.ref_yaw)), self.ref_yaw
        assert np.all(~np.isnan(self.ref_k)), self.ref_k
        assert np.all(~np.isnan(self.ref_delta)), self.ref_delta

    def get_ref(self, start, N):
        """
        :param start: 起始索引
        :param N: 预测步数
        """
        end = min(start + N + 1, len(self.ref_x))
        x_ref = np.vstack((self.ref_x[start:end], self.ref_y[start:end],
                           self.ref_yaw[start:end], self.ref_v[start:end]))
        u_ref = np.vstack((self.ref_a[start:end], self.ref_delta[start:end]))
        return x_ref, u_ref

    def get_ref_pos(self, start, time_len):
        """
        :param start: 起始索引
        :param time_len: 预测时间长度
        """
        N = int(time_len / self.dt)
        end = min(start + N + 1, len(self.ref_x))
        x_ref = np.vstack((self.ref_x[start:end], self.ref_y[start:end],
                           self.ref_yaw[start:end], self.ref_v[start:end], self.ref_a[start:end])).T
        traj_point = []
        for x, y, yaw, v, a in x_ref:
            traj_point.append(TrajPoint(
                x=x,
                y=y,
                acc=a,
                speed=v,
                yaw=yaw,
            ))
        addi_num = N + 1 - len(traj_point)
        if addi_num > 0:
            for i in range(addi_num):
                traj_point.append(TrajPoint(
                    x=traj_point[-1].x + traj_point[-1].vx * self.dt,
                    y=traj_point[-1].y + traj_point[-1].vy * self.dt,
                    acc=traj_point[-1].acc,
                    speed=traj_point[-1].speed,
                    yaw=traj_point[-1].yaw,
                ))
        return traj_point

    def get_index(self, start, distance):
        """
        根据起始索引和前视距离获取索引

        :param start: 起始索引
        :param distance: 前视距离
        """
        length_cum = 0
        if self.ref_len is None:
            pos_len = np.linalg.norm(np.vstack((np.diff(self.ref_x), np.diff(self.ref_y))), axis=0)
            pos_len = np.insert(pos_len, 0, 0)
            self.ref_len = np.cumsum(pos_len)
        current_len = self.ref_len[start]
        target_index = np.where(self.ref_len > current_len + distance)[0]
        if len(target_index) == 0:
            return len(self.ref_x) - 1
        return target_index[0]
