# -*- coding = uft-8 -*-
# @time : 2022/1/11
# @Author : yzbyx
# @File : obstacle.py
# @Software : PyCharm
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from trasim_simplified.core.constant import COLOR, TrajPoint, V_TYPE


class Obstacle:
    def __init__(self, type_: V_TYPE):
        self.x = 0
        """车头中点的x坐标 [m]"""
        self.y = 0
        """车头中点的y坐标 [m]"""
        self.speed = 0
        """车辆速度 [m/s]"""
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

        self.wb_prop = 0.6
        self._update_prop()
        self.l_rear_axle_2_head = self.wheelbase * ((1 - self.wb_prop) / 2 + self.wb_prop)
        self._define_shape()

    @property
    def wheelbase(self):
        return self.length * self.wb_prop

    @property
    def v(self) -> float:
        return self.speed * np.cos(self.yaw)

    @property
    def v_lat(self):
        return self.speed * np.sin(self.yaw)

    @property
    def a(self):
        return self.acc * np.cos(self.yaw)

    @property
    def a_lat(self):
        return self.acc * np.sin(self.yaw)

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

    def get_ctrl(self):
        return self.acc, self.delta

    def get_state(self):
        return self.x, self.y, self.yaw, self.speed

    def _define_shape(self):
        """
        定义朝向x轴正方向，车头在前，车尾在后的车身和车轮轮廓
        """
        WIDTH = 1.8
        WHEEL_LEN = 0.8
        WHEEL_WIDTH = 0.2
        TREAD = 0.6
        self.outline_poses = np.array(
            [(- self.length / 2, - WIDTH / 2),
             (- self.length / 2, WIDTH / 2),
             (self.length / 2, WIDTH / 2),
             (self.length / 2, - WIDTH / 2),
             (- self.length / 2, - WIDTH / 2)]
        )
        self.rect_kwargs = {"fill": False, "edgecolor": "black"}
        # rear right wheel
        self.rr_wheel_poses = np.array(
            [(- self.wheelbase / 2 - WHEEL_LEN / 2, - WHEEL_WIDTH / 2 - TREAD),
             (- self.wheelbase / 2 - WHEEL_LEN / 2, WHEEL_WIDTH / 2 - TREAD),
             (- self.wheelbase / 2 + WHEEL_LEN / 2, WHEEL_WIDTH / 2 - TREAD),
             (- self.wheelbase / 2 + WHEEL_LEN / 2, - WHEEL_WIDTH / 2 - TREAD),
             (- self.wheelbase / 2 - WHEEL_LEN / 2, - WHEEL_WIDTH / 2 - TREAD)]
        )
        # rear left wheel
        self.rl_wheel_poses = self.rr_wheel_poses + np.array([0, 2 * TREAD])
        # front right wheel
        self.fr_wheel_poses = self.rr_wheel_poses + np.array([self.wheelbase, 0])
        # front left wheel
        self.fl_wheel_poses = self.fr_wheel_poses + np.array([0, 2 * TREAD])

    def get_traj_point(self, is_center=False):
        if is_center:
            return TrajPoint(
                x=self.x_c,
                y=self.y_c,
                yaw=self.yaw,
                speed=self.speed,
                acc=self.acc,
                delta=self.delta,
                length=self.length,
                width=self.width
            )
        return TrajPoint(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            speed=self.speed,
            acc=self.acc,
            delta=self.delta,
            length=self.length,
            width=self.width
        )

    def get_state_for_traj(self):
        """返回车辆状态量[x, dx, ddx, y, dy, ddy]"""
        return np.array([self.x, self.v, self.a, self.y, self.v_lat, self.a_lat])

    def update_state(self, a, delta):
        """
        Update ego_local_state

        :param a: acceleration
        :param delta: steering angle
        """
        if abs((a - self.acc) / self.dt) > self.JERK_MAX:
            a = self.acc + self.dt * self.JERK_MAX * np.sign(a - self.acc)

        if abs((delta - self.delta) / self.dt) > self.D_DELTA_MAX:
            delta = self.delta + self.dt * self.D_DELTA_MAX * np.sign(delta - self.delta)

        self.acc = a
        self.delta = delta

        self.speed = self.speed + self.dt * a
        if self.speed < 0:
            self.speed = 0
        # head
        beta = np.arctan(self.prop_ * np.tan(delta))
        self.x = self.x + self.dt * self.speed * np.cos(self.yaw + beta)
        self.y = self.y + self.dt * self.speed * np.sin(self.yaw + beta)
        self.yaw = self.yaw + self.dt * self.speed * np.sin(beta) / (self.wheelbase * self.prop_)

        # if np.isnan(self.x):
        #     print(self.speed, self.yaw, self.yaw + beta)

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
        model = self.clone()
        for (ai, di, i) in zip(oa, od, range(1, N + 1)):
            model.update_state(ai, di)
            xbar[0, i] = model.x
            xbar[1, i] = model.y
            xbar[2, i] = model.yaw
            xbar[3, i] = model.speed

        return xbar

    def clone(self):
        """
        克隆当前对象
        """
        new_obj = Obstacle(self.type)
        new_obj.x = self.x
        new_obj.y = self.y
        new_obj.speed = self.speed
        new_obj.acc = self.acc
        new_obj.yaw = self.yaw
        new_obj.delta = self.delta
        new_obj.length = self.length
        new_obj.width = self.width
        new_obj.dt = self.dt
        return new_obj

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

    def get_bbox(self, five_points=False):
        yaw = self.yaw
        # 以车头中心为原点
        if five_points:
            outline = np.array(
                [(- self.length, - self.width / 2),
                    (- self.length, self.width / 2),
                    (0, self.width / 2),
                    (0, - self.width / 2),
                    (- self.length / 2, - self.width / 2)]
            )
        else:
            outline = np.array(
                [(- self.length, - self.width / 2),
                 (- self.length, self.width / 2),
                 (0, self.width / 2),
                 (0, - self.width / 2)]
            )
        yaw_rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        outline = np.dot(yaw_rot, outline.T).T
        outline += np.array([self.x, self.y])
        return outline

    def plot_car(self, ax: plt.Axes):
        x, y, yaw = self.x_c, self.y_c, self.yaw
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
        # 填充颜色
        ax.fill(total[0][:, 0], total[0][:, 1], color=normalize_color(self.color), alpha=0.5)

        return patches


def normalize_color(color: Tuple[int, int, int]) -> tuple[float, ...]:
    """将0-255范围的RGB颜色值转换为0-1范围"""
    return tuple(c / 255.0 for c in color)
