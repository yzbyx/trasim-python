# -*- coding: utf-8 -*-
# @time : 2025/4/14 14:55
# @Author : yzbyx
# @File : mpc_test.py
# Software: PyCharm

import numpy as np
import cvxopt

# 自行车模型参数
L = 2.0  # 轴距，单位：米

# MPC参数
N = 20  # 时间步长数
dt = 0.1  # 每一步的时间间隔，单位：秒
Q = np.diag([1, 1, 0.1, 0.1])  # 状态误差惩罚
R = np.diag([0.1, 0.1])  # 控制输入惩罚

# 初始状态
x_init = np.array([0, 0, 0, 0])  # 初始位置 (x, y) 和朝向角度以及速度

# 给定轨迹（示例路径）
path = np.array([[i, np.sin(i)] for i in np.linspace(0, 10, 100)])

# 自行车运动模型（单轨模型）
def bicycle_model(x, u, dt):
    # x = [x, y, theta, v] (位置、速度、朝向角度)
    # u = [a, delta] (加速度、转向角)

    x1, y1, theta1, v1 = x
    a, delta = u
    theta = theta1 + delta
    v = v1 + a * dt
    x2 = x1 + v * np.cos(theta) * dt
    y2 = y1 + v * np.sin(theta) * dt

    return np.array([x2, y2, theta, v])

# 目标轨迹跟踪
def mpc_control(x_init, path):
    # 设定优化变量
    x = np.zeros((N+1, 4))  # 状态变量（N+1个时刻，4维状态：x, y, theta, v）
    u = np.zeros((N, 2))  # 控制输入（N个时刻，2维输入：a, delta）

    # 目标函数：状态和控制输入的误差
    def objective(u, x, path):
        cost = 0
        for k in range(N):
            # 当前状态与目标轨迹的偏差
            target = path[k] if k < len(path) else path[-1]
            dx = x[k, 0] - target[0]
            dy = x[k, 1] - target[1]
            cost += np.dot(np.array([dx, dy, x[k, 2], x[k, 3]]).T, np.dot(Q, np.array([dx, dy, x[k, 2], x[k, 3]])))

            # 控制输入的惩罚
            cost += np.dot(u[k, :], np.dot(R, u[k, :].T))

        return cost

    # 动态约束：状态更新
    def dynamics_constraints(x, u):
        cons = []
        for k in range(N):
            x_next = bicycle_model(x[k], u[k], dt)
            cons.append(x[k+1] - x_next)
        return cons

    # 约束：输入限制（加速度和转向角度）
    def input_constraints(u):
        cons = []
        for k in range(N):
            cons.append(u[k, 0] - 1.0)  # 加速度最大为1 m/s^2
            cons.append(u[k, 0] + 1.0)  # 加速度最小为-1 m/s^2
            cons.append(u[k, 1] - np.pi/6)  # 转向角最大为30度
            cons.append(u[k, 1] + np.pi/6)  # 转向角最小为-30度
        return cons

    # 构造优化问题
    H = np.zeros((N*2, N*2))  # 二次项
    f = np.zeros(N*2)  # 线性项

    # 初始状态
    x[0] = x_init

    # 优化约束
    G = np.zeros((2*N, N*2))  # 约束矩阵
    h = np.zeros(2*N)  # 约束上限
    A = np.zeros((N, N*2))  # 目标约束

    # 优化求解器
    P = cvxopt.matrix(H)
    q = cvxopt.matrix(f)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(np.zeros(N))

    # 求解QP问题
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)

    # 解出控制输入u
    u_opt = np.array(sol['x'])[:N, :]
    return u_opt

# 调用MPC控制器进行轨迹跟踪
u_optimal = mpc_control(x_init, path)

# 输出最优控制输入
print("Optimal control inputs (a, delta):")
print(u_optimal)
