# -*- coding: utf-8 -*-
# @Time : 2024/12/18 16:30
# @Author : yzbyx
# @File : ref_path.py
# Software: PyCharm
import numpy as np

from trasim_simplified.core.agent.vehicle import Vehicle

np.seterr(divide='ignore', invalid='ignore')


class ReferencePath:
    def __init__(self, ref_path, dt):
        self.ref_x = ref_path[:, 0]
        self.ref_dx = ref_path[:, 1] / dt
        self.ref_ddx = ref_path[:, 2] / (dt ** 2)
        self.ref_y = ref_path[:, 3]
        self.ref_dy = ref_path[:, 4] / dt
        self.ref_ddy = ref_path[:, 5] / (dt ** 2)

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

        # 速度和加速度为后向差分
        self.ref_v = np.linalg.norm(np.vstack((self.ref_dx, self.ref_dy)), axis=0)  # 参考速度
        self.ref_a = np.linalg.norm(np.vstack((self.ref_ddx, self.ref_ddy)), axis=0)  # 参考加速度

        self.ref_yaw = np.arctan2(self.ref_dy, self.ref_dx)  # 参考航向角

        # kappa = r'(t) x r''(t) / |r'(t)|^3
        self.ref_k = ((self.ref_ddy * self.ref_dx - self.ref_ddx * self.ref_dy) / (
                (self.ref_dx ** 2 + self.ref_dy ** 2) ** (3 / 2))
             )  # 曲率k计算

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
