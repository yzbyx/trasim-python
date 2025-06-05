# -*- coding: utf-8 -*-
# @time : 2024/12/19 15:44
# @Author : yzbyx
# @File : mpc_solver.py
# Software: PyCharm
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import tqdm
from matplotlib import pyplot as plt

from traj_process.util.plot_helper import get_fig_ax
from trasim_simplified.core.agent.ref_path import ReferencePath

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle


class MPC_Solver:
    def __init__(self, N_MPC, ref_path: ReferencePath, ugv: 'Vehicle'):
        """
        MPC Solver

        :param N_MPC: prediction horizon
        """
        self.dynamic_n_mpc = False
        self.N_MPC = N_MPC
        self.ref_path = ref_path
        self.ref_path.cal_ref()
        self.ref_path.cal_ref_delta(ugv)
        self.ugv = ugv
        self.step = 0

        self.opt_x = None
        self.opt_u = None

        self.Q = np.diag([10, 10, 10, 10])  # state cost matrix
        self.Qf = self.Q * 100  # state final matrix
        self.use_R = False
        self.use_Rd = False
        self.use_Q = True
        self.use_Qu = False
        self.use_Qf = True

        # self.R = np.diag([0.1, 1])  # input cost matrix
        # self.Rd = np.diag([0.1, 1])  # input diff cost matrix
        # self.Q = np.diag([0.1, 0.1, 1, 1])  # state cost matrix
        # self.Qu = np.diag([0.01, 1])  # input-ref diff cost matrix
        # self.Qf = self.Q * 10  # state final matrix
        # self.use_R = True
        # self.use_Rd = True
        # self.use_Q = True
        # self.use_Qu = True
        # self.use_Qf = True

        self.total_j = 0
        self.return_j = False
        self.is_end = False

    def init_mpc(self, dynamic_n_mpc=False, print_config=False, return_j=False):
        self.return_j = return_j
        self.dynamic_n_mpc = dynamic_n_mpc
        if print_config:
            self.print_config()

    def step_mpc(self):
        if not self.return_j:
            u, x, x_bar, x_ref, is_end = self.mpc_prepare(zero_ref_a=False)
        else:
            u, x, x_bar, x_ref, is_end, j = self.mpc_prepare(zero_ref_a=False, return_j=True)
            self.total_j += j

        # self.ugv.update_state(u[0, 1], u[1, 1])

        self.step += 1

        self.is_end = is_end

        return u[0, 1], u[1, 1], is_end

    def mpc_prepare(self, zero_ref_a=False, return_j=False):
        # 1s后的index
        end = min(self.step + round(1 / self.ugv.dt), len(self.ref_path.ref_v) - 1)
        # 是否到达终点
        is_end = True if (end == len(self.ref_path.ref_v) - 1) else False

        if self.dynamic_n_mpc:
            # 动态N_MPC
            current_v = self.ugv.get_state()[3]

            T_desire = 1.6
            dist = max(10, current_v * T_desire)  # 1.6s前视距离（IDM）
            end = self.ref_path.get_index(self.step, dist)
            self.N_MPC = end - self.step

        x_ref, u_ref = self.ref_path.get_ref(self.step, self.N_MPC)

        u_ref[:, 0] = self.ugv.get_ctrl()
        x_ref[:, 0] = self.ugv.get_state()

        if zero_ref_a:
            u_ref[0, :] = 0

        # u_ref = np.zeros(u_ref.shape)
        # u_ref[1, :] = 0

        x0 = self.ugv.get_state()
        x_bar_first = None
        x_bar = self.ugv.predict_motion(x0, u_ref[:, 1:])
        # print("u_ref", u_ref)
        # print("x_bar", x_bar)
        # print("x_ref", x_ref)
        # print("x_delta", x_ref - x_bar)

        if x_bar_first is None:
            x_bar_first = x_bar
        x_val, u_val, j = self.mpc_control(x_ref, x_ref, u_ref)

        # x_bar_bar = self.ugv.predict_motion(x0, u_val[:, 1:])

        # fig, ax = get_fig_ax()
        # ax.plot(x_ref[0, :], x_ref[1, :], label="ref")
        # ax.plot(x_bar[0, :], x_bar[1, :], label="bar")
        # ax.plot(x_bar_bar[0, :], x_bar_bar[1, :], label="val")
        # ax.legend()
        # print(u_ref)
        # plt.ioff()
        # plt.show()
        # plt.ion()

        if return_j:
            return u_val, x_val, x_bar_first, x_ref, is_end, j
        return u_val, x_val, x_bar_first, x_ref, is_end

    def mpc_control(self, x_ref, x_bar, u_ref):
        """
        linear mpc control

        :param x_ref: reference states trajectory
        :param x_bar: predicted states trajectory
        :param u_ref: reference control trajectory
        """
        cost, constraints, x, u = self.get_cost_and_constrains(x_ref, x_bar, u_ref)

        x.value = x_bar
        u.value = u_ref

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.GUROBI, verbose=False)
        except cp.error.SolverError:
            prob.solve(solver=cp.OSQP, verbose=False)

        statu = prob.status == cp.OPTIMAL
        x_val = x.value
        u_val = u.value
        j = prob.value

        if not statu:
            print(statu, f"Optimal solution not found: {prob.status}")

        return x_val, u_val, j

    def get_cost_and_constrains(self, x_ref, x_bar, u_ref):
        cost = 0  # 代价函数
        constraints = []  # 约束条件

        assert x_ref.shape[1] == self.N_MPC + 1, print(x_ref)

        x = cp.Variable((self.ugv.NX, self.N_MPC + 1))  # 当前状态 + N_MPC个预测状态
        u = cp.Variable((self.ugv.NU, self.N_MPC + 1))  # 当前控制 + N_MPC个预测控制

        assume_PSD = True

        for i, t in enumerate(range(self.N_MPC)):
            cost += cp.quad_form(u[:, t + 1], self.R, assume_PSD=assume_PSD) if self.use_R else 0

            if t != self.N_MPC - 1:
                cost += cp.quad_form(x[:, t + 1] - x_ref[:, t + 1], self.Q, assume_PSD=assume_PSD) if self.use_Q else 0
            else:
                cost += cp.quad_form(x[:, self.N_MPC] - x_ref[:, self.N_MPC], self.Qf, assume_PSD=assume_PSD) \
                    if self.use_Qf else 0

            cost += cp.quad_form(x[:, t + 1] - x_ref[:, t + 1], self.Q, assume_PSD=assume_PSD) if self.use_Q else 0
            cost += cp.quad_form(u[:, t + 1] - u_ref[:, t + 1], self.Qu, assume_PSD=assume_PSD) if self.use_Qu else 0
            cost += cp.quad_form(u[:, t + 1] - u[:, t], self.Rd, assume_PSD=assume_PSD) if self.use_Rd else 0

            A, B = self.ugv.get_error_state_space(x_bar[3, t], x_bar[2, t], u_ref[1, t + 1])
            constraints += [
                x[:, t + 1] - x_bar[:, t + 1] == A @ (x[:, t] - x_bar[:, t]) + B @ (u[:, t + 1] - u_ref[:, t + 1])
            ]

            # 速度限制
            constraints += [x[3, t + 1] >= 0]  # 速度非负
            constraints += [x[3, t + 1] <= self.ugv.V_MAX]  # 速度限制
            # constraints += [cp.abs(x_ref[0, t + 1] - x[0, t + 1]) <= 0.1]  # Tube-MPC鲁棒约束
            # constraints += [cp.abs(x_ref[1, t + 1] - x[1, t + 1]) <= 0.1]

        # 初始状态约束
        constraints += [x[:, 0] == x_ref[:, 0]]
        constraints += [u[:, 0] == u_ref[:, 0]]

        # 加速度限制
        constraints += [cp.norm(u[0, :], "inf") <= self.ugv.A_MAX]
        constraints += [cp.norm(cp.diff(u[0, :]), "inf") <= self.ugv.JERK_MAX * self.ugv.dt]

        # 前轮转角限制
        constraints += [cp.norm(u[1, :], "inf") <= self.ugv.DELTA_MAX]
        constraints += [cp.norm(cp.diff(u[1, :]), "inf") <= self.ugv.D_DELTA_MAX * self.ugv.dt]
        # constraints += [cp.norm(cp.diff(u[1, :], 2), "inf") <= self.ugv.DD_DELTA_MAX * self.ugv.dt ** 2]

        return cost, constraints, x, u

    def print_config(self):
        """
        Print MPC config
        """
        config_str = (f"dynamic_n_mpc: {self.dynamic_n_mpc}\n"
                      f"R({self.use_R}): {[float(self.R[i, i]) for i in range(2)]}\n"
                      f"Rd({self.use_Rd}): {[float(self.Rd[i, i]) for i in range(2)]}\n"
                      f"Q({self.use_Q}): {[float(self.Q[i, i]) for i in range(4)]}\n"
                      f"Qu({self.use_Qu}): {[float(self.Qu[i, i]) for i in range(2)]}\n"
                      f"Qf({self.use_Qf}): {[float(self.Qf[i, i]) for i in range(4)]}\n")
        print(config_str)
