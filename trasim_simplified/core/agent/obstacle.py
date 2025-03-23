# -*- coding = uft-8 -*-
# @Time : 2022/1/11
# @Author : yzbyx
# @File : obstacle.py
# @Software : PyCharm
import numpy as np
from matplotlib import pyplot as plt

from trasim_simplified.core.constant import COLOR


class Obstacle:
    def __init__(self, type_: int):
        self.x = 0
        """车头中点的x坐标 [m]"""
        self.y = 0
        """车头中点的y坐标 [m]"""

        self.v = 0
        """车辆纵向速度 [m/s]"""
        self.v_lat = 0
        """车辆横向速度 [m/s]"""
        self.speed = 0
        """车辆速度 [m/s]"""

        self.a = 0
        """车辆纵向加速度 [m/s^2]"""
        self.a_lat = 0
        """车辆横向加速度 [m/s^2]"""
        self.acc = 0
        """车辆加速度 [m/s^2]"""

        self.yaw = 0
        self.delta = 0

        self.color = COLOR.yellow

        self.length = 5.0
        self.width = 1.8
        self.type = type_

        self.NX = 4
        self.NU = 2

        self.dt = 0.1

        self.V_MAX = 40
        self.A_MAX = 11.5
        self.JERK_MAX = 15
        self.DELTA_MAX = 1.066
        self.D_DELTA_MAX = 0.4
        self.DD_DELTA_MAX = 20

        self.length = 5
        self.wb_prop = 0.6
        self.wheelbase = self.length * self.wb_prop
        self._update_prop()
        self.l_rear_axle_2_head = self.wheelbase * ((1 - self.wb_prop) / 2 + self.wb_prop)
        self._define_shape()

    @property
    def position(self):
        return np.array([self.x, self.y])

    @property
    def velocity(self):
        return np.array([self.v, self.v_lat])

    @property
    def x_c(self):
        return self.x - 0.5 * self.length * np.cos(self.yaw)

    @property
    def y_c(self):
        return self.y - 0.5 * self.length * np.sin(self.yaw)

    def _update_prop(self):
        self.prop_ = ((1 - self.wb_prop) / 2 + self.wb_prop) / self.wb_prop
        """ the wb_prop of length of rear wheel to car head to the wheelbase """

    def set_state(self, x, y, yaw, v):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = v

    def set_ctrl(self, a, delta):
        self.acc = a
        self.delta = delta

    def _define_shape(self):
        """
        定义朝向x轴正方向，车头在前，车尾在后的车身和车轮轮廓
        """
        WIDTH = 1.8
        LENGTH = self.wheelbase / self.wb_prop
        BACK_TO_WHEEL = LENGTH * (1 - self.wb_prop) / 2
        WHEEL_LEN = 0.8
        WHEEL_WIDTH = 0.2
        TREAD = 0.6
        self.outline_poses = np.array(
            [(- BACK_TO_WHEEL, - WIDTH / 2),
             (- BACK_TO_WHEEL, WIDTH / 2),
             (self.wheelbase * self.prop_, WIDTH / 2),
             (self.wheelbase * self.prop_, - WIDTH / 2),
             (- BACK_TO_WHEEL, - WIDTH / 2)]
        )
        self.rect_kwargs = {"fill": False, "edgecolor": "black"}
        # following right wheel
        self.rr_wheel_poses = np.array(
            [(- WHEEL_LEN / 2, - WHEEL_WIDTH / 2 - TREAD),
             (- WHEEL_LEN / 2, WHEEL_WIDTH / 2 - TREAD),
             (WHEEL_LEN / 2, WHEEL_WIDTH / 2 - TREAD),
             (WHEEL_LEN / 2, -WHEEL_WIDTH / 2 - TREAD),
             (- WHEEL_LEN / 2, - WHEEL_WIDTH / 2 - TREAD)]
        )
        # following left wheel
        self.rl_wheel_poses = self.rr_wheel_poses + np.array([0, 2 * TREAD])
        # front right wheel
        self.fr_wheel_poses = self.rr_wheel_poses + np.array([self.wheelbase, 0])
        # front left wheel
        self.fl_wheel_poses = self.fr_wheel_poses + np.array([0, 2 * TREAD])

    def update_state(self, a, delta):
        """
        Update ego_local_state

        :param a: acceleration
        :param delta: steering angle
        """
        self.acc = a
        self.delta = delta

        self.speed = self.speed + self.dt * a
        # head
        beta = np.arctan(self.prop_ * np.tan(delta))
        self.x = self.x + self.dt * self.speed * np.cos(self.yaw + beta)
        self.y = self.y + self.dt * self.speed * np.sin(self.yaw + beta)
        self.yaw = self.yaw + self.dt * self.speed * np.sin(beta) / (self.wheelbase * self.prop_)

    def predict_motion(self, x0, u_ctrl):
        """
        Predict motion by input

        :param x0: initial ego_local_state
        :param u_ctrl: control input

        :return: predicted ego_local_state (contain initial ego_local_state)
        """
        N = np.shape(u_ctrl)[1]
        xbar = np.zeros((self.NX, N + 1))
        for i in range(len(x0)):
            xbar[i, 0] = x0[i]

        oa = u_ctrl[0, :]
        od = u_ctrl[1, :]
        model = Obstacle(self.type)
        for (ai, di, i) in zip(oa, od, range(1, N + 1)):
            model.update_state(ai, di)
            xbar[0, i] = model.x
            xbar[1, i] = model.y
            xbar[2, i] = model.yaw
            xbar[3, i] = model.speed

        return xbar

    def get_error_state_space(self, ref_v, ref_yaw, ref_delta):
        """
        误差状态空间方程
        """
        # 以车头为中心
        ref_beta = np.arctan(self.prop_ * np.tan(ref_delta))
        # [x, y, yaw, v], [a, delta]
        A = np.array([
            # x_dot = V * cos(yaw + beta)
            [1, 0, -ref_v * self.dt * np.sin(ref_yaw + ref_beta), self.dt * np.cos(ref_yaw + ref_beta)],
            # y_dot = V * sin(yaw + beta)
            [0, 1, ref_v * self.dt * np.cos(ref_yaw + ref_beta), self.dt * np.sin(ref_yaw + ref_beta)],
            # yaw_dot = V * sin(beta) / (wheelbase * prop_)
            [0, 0, 1, np.sin(ref_beta) / (self.wheelbase * self.prop_) * self.dt],
            # v_dot = A
            [0, 0, 0, 1]
        ])
        c = 1 / (1 + (self.prop_ * np.tan(ref_delta)) ** 2) * (1 / (2 * np.cos(ref_delta) ** 2))
        B = np.array([
            [0, -ref_v * np.sin(ref_yaw + ref_beta) * c * self.dt],
            [0, ref_v * np.cos(ref_yaw + ref_beta) * c * self.dt],
            [0, ref_v * np.cos(ref_beta) * c / (self.wheelbase * self.prop_) * self.dt],
            [self.dt, 0]
        ])

        return A, B

    def get_bbox(self):
        yaw = self.yaw
        outline = np.copy(self.outline_poses)
        yaw_rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        outline = np.dot(yaw_rot, outline.T).T
        return outline

    def plot_car(self, ax: plt.Axes):
        x, y, yaw = self.x, self.y, self.yaw
        steer = self.delta

        outline = np.copy(self.outline_poses)
        rr_wheel = np.copy(self.rr_wheel_poses)
        rl_wheel = np.copy(self.rl_wheel_poses)
        fr_wheel = np.copy(self.fr_wheel_poses)
        fl_wheel = np.copy(self.fl_wheel_poses)
        total = [outline, rr_wheel, rl_wheel, fr_wheel, fl_wheel]

        yaw_rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        delta_rot = np.array([[np.cos(steer), -np.sin(steer)], [np.sin(steer), np.cos(steer)]])
        for i, item in enumerate(total):
            total[i] = np.dot(yaw_rot, item.T).T
        front_wheel = total[3:]
        for i, item in enumerate(front_wheel):
            center = item.mean(axis=0)
            item -= center
            item = np.dot(delta_rot, item.T).T
            total[3 + i] = item + center

        for i, item in enumerate(total):
            item += np.array([x, y])
        patches = [
            ax.plot(item[:, 0], item[:, 1], "k-")[0] for item in total
        ]

        return patches
