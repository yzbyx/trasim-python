# -*- coding: utf-8 -*-
# @Time : 2025/5/12 21:20
# @Author : yzbyx
# @File : opti_quintic.py
# Software: PyCharm
from typing import TYPE_CHECKING, Optional

import cvxpy
import numpy as np

from trasim_simplified.core.agent.utils import get_y_guess
from trasim_simplified.core.constant import GameVehSurr, SolveRes

if TYPE_CHECKING:
    from trasim_simplified.core.agent.game_agent import Game_A_Vehicle, Game_Vehicle
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract

A = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # dx0
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ddx0
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # y0
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # dy0
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # ddy0
])


def get_A2(T):
    return np.array([
        [1, T, T ** 2, T ** 3, T ** 4, T ** 5],  # y1
        [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],  # dy1
        [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]  # ddy1
    ])


def get_traj_own_constraints_and_costs(
        x: cvxpy.Variable, veh: 'Game_A_Vehicle',
        target_lane: 'LaneAbstract', T
) -> tuple[list[cvxpy.Constraint], cvxpy.Expression, cvxpy.Expression, float]:
    y1 = target_lane.y_center
    dt = veh.dt
    A2 = get_A2(T)
    b2 = np.array([y1, 0, 0])
    time_steps = np.arange(0, T + dt / 2, dt)

    constraints = []

    # 初始与终止状态约束
    constraints += [A @ x[: 12] == veh.state]  # 初始状态约束
    constraints += [A2 @ x[6: 12] == b2]  # 终止状态约束

    # 计算每个时间步的y约束
    A_xy = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])
    if target_lane.index > veh.lane.index:
        constraints += [A_xy @ x[6: 12] <= veh.lane.y_left]
        constraints += [A_xy @ x[6: 12] <= target_lane.y_right]
    else:
        constraints += [A_xy @ x[6: 12] <= target_lane.y_left]
        constraints += [A_xy @ x[6: 12] <= veh.lane.y_right]

    # 计算每个时间步的dx约束
    A_dxy = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps])
    constraints += [A_dxy @ x[: 6] >= 0]
    constraints += [A_dxy @ x[: 6] <= veh.vel_desire + 5]

    # 计算每个时间步的ddx和ddy约束
    A_ddxy = np.array([[0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3] for t in time_steps])
    b_ineq5 = np.array([veh.acc_desire] * len(A_ddxy))
    b_ineq6 = np.array([- veh.dec_desire] * len(A_ddxy))
    constraints += [A_ddxy @ x[6: 12] <= b_ineq5]
    constraints += [A_ddxy @ x[6: 12] >= b_ineq6]
    constraints += [A_ddxy @ x[: 6] <= b_ineq5]
    constraints += [A_ddxy @ x[: 6] >= b_ineq6]

    # 舒适成本
    A_xy_jerk = np.array([[0, 0, 0, 6, 24 * t, 60 * t ** 2] for t in time_steps])
    com_cost = 0.5 * cvxpy.max(cvxpy.abs(A_xy_jerk @ x[:6])) / veh.JERK_MAX
    com_cost += 0.5 * cvxpy.max(cvxpy.abs(A_xy_jerk @ x[6:12])) / veh.JERK_MAX
    # com_cost = cvxpy.min(cvxpy.hstack([[1], com_cost]))
    # com_cost = cvxpy.max(cvxpy.hstack([[-1], com_cost]))

    # 效率成本
    ev_v = A_dxy @ x[:6]
    eff_cost = (ev_v[0] - cvxpy.mean(ev_v)) / 10
    # eff_cost = cvxpy.min(cvxpy.hstack([[1], eff_cost]))
    # eff_cost = cvxpy.max(cvxpy.hstack([[-1], eff_cost]))
    eff_cost += (T - 5) / 5

    # 路径成本
    lc_direction = target_lane.index - veh.lane.index
    route_incentive = veh.route_incentive(lc_direction)
    route_cost = - route_incentive

    return constraints, com_cost, eff_cost, route_cost


def get_single_surr_constraints_and_costs(
        veh: 'Game_A_Vehicle',
        ev_pos, ev_v, other_pos, other_v, other_l, is_preceding
):
    x_relax: cvxpy.Variable = cvxpy.Variable(1)
    v = ev_v if is_preceding else other_v
    sign = 1 if is_preceding else -1
    l = other_l if is_preceding else veh.length

    constraints = []

    other_pos = other_pos - l * sign
    b_ineq_part1 = other_pos - (v * veh.time_safe + veh.safe_s0) * veh.SCALE * sign
    constraints += [(ev_pos - b_ineq_part1) * sign <= 0]
    b_ineq_part2 = other_pos - (v * veh.time_wanted + veh.safe_s0) * veh.SCALE * sign
    constraints += [x_relax >= (ev_pos - b_ineq_part2) * sign / (veh.state[1] * veh.time_wanted * veh.SCALE)]

    cost = cvxpy.maximum(x_relax, -1)
    # cost = cvxpy.minimum(cost, 1)

    return constraints, cost


def get_lc_cross_step(T, veh: 'Game_A_Vehicle', target_lane: 'LaneAbstract', ori_lane: 'LaneAbstract'):
    time_steps = np.arange(0, T + veh.dt / 2, veh.dt)
    y, yf, dyf, ddyf = get_y_guess(T, veh.state, target_lane.y_center)
    A_y = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])
    Y = A_y @ y
    # 进入目标车道的初始时刻
    if Y[0] < Y[-1]:
        step = (Y[Y <= ori_lane.y_left]).shape[0]  # 进入目标车道的初始时刻
    else:
        step = (Y[Y >= ori_lane.y_right]).shape[0]  # 进入目标车道的初始时刻
    return step


def get_surr_constraints_and_costs(
        x: cvxpy.Variable, veh: 'Game_A_Vehicle', T: float,
        TP: 'Game_Vehicle', TR: 'Game_Vehicle', PC: 'Game_Vehicle', CR: 'Game_Vehicle',
        TP_traj: np.ndarray, TR_traj: np.ndarray, PC_traj: np.ndarray, CR_traj: np.ndarray,
        target_lane: 'LaneAbstract', ori_lane: 'LaneAbstract',
):
    """
    计算周边车辆影响下的换道约束
    :param x: 横纵五次多项式系数+松弛变量
    :param veh: 当前车
    :param T: 持续时间
    :param TP: 目标间隙前车
    :param TR: 目标间隙后车（交织换道情况下为TRR）
    :param PC: 初始车道前车
    :param CR: 初始车道后车（队列换道下为CRR）
    :param TP_traj: 目标间隙前车
    :param TR_traj: 目标间隙后车（交织换道情况下为TRR）
    :param PC_traj: 初始车道前车
    :param CR_traj: 初始车道后车（队列换道下为CRR）
    :param target_lane: 目标车道
    :param ori_lane: 初始车道
    :return:
    """
    xt_TP, dxt_TP = TP_traj[:, 0], TP_traj[:, 1]
    xt_TR, dxt_TR = TR_traj[:, 0], TR_traj[:, 1]
    xt_PC, dxt_PC = PC_traj[:, 0], PC_traj[:, 1]
    xt_CR, dxt_CR = CR_traj[:, 0], CR_traj[:, 1]

    l_TP = TP.length
    l_TR = TR.length
    l_PC = PC.length
    l_CR = CR.length

    constraints = []

    dt = veh.dt
    time_steps = np.arange(0, T + dt / 2, dt)
    dxt = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps])
    A_xy = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])  # 纵向位置
    ev_pos = A_xy @ x[: 6]  # 纵向位置
    ev_v = dxt @ x[:6]

    if veh.lane == target_lane:
        cp_constraints, cp_cost = get_single_surr_constraints_and_costs(
            veh, ev_pos, ev_v, xt_PC, dxt_PC, l_PC, True
        )

        cr_constraints, cr_cost = get_single_surr_constraints_and_costs(
            veh, ev_pos, ev_v, xt_CR, dxt_CR, l_CR, False
        )

        tr_cost = 0
        tp_cost = 0
        step = None

        constraints += cp_constraints
        constraints += cr_constraints
    else:
        step = get_lc_cross_step(T, veh, target_lane, ori_lane)

        cp_constraints, cp_cost = get_single_surr_constraints_and_costs(
            veh, ev_pos[:step], ev_v[:step], xt_PC[:step], dxt_PC[:step], l_PC, True
        )
        cr_constraints, cr_cost = get_single_surr_constraints_and_costs(
            veh, ev_pos[:step], ev_v[:step], xt_CR[:step], dxt_CR[:step], l_CR, False
        )
        tp_constraints, tp_cost = get_single_surr_constraints_and_costs(
            veh, ev_pos[step:], ev_v[step:], xt_TP[step:], dxt_TP[step:], l_TP, True
        )
        tr_constraints, tr_cost = get_single_surr_constraints_and_costs(
            veh, ev_pos[step:], ev_v[step:], xt_TR[step:], dxt_TR[step:], l_TR, False
        )

        constraints += cp_constraints
        constraints += cr_constraints
        constraints += tp_constraints
        constraints += tr_constraints

    # 安全成本
    safe_cost = cvxpy.max(cvxpy.hstack([cp_cost, cr_cost, tp_cost, tr_cost]))

    return constraints, safe_cost, step


def solve_quintic_given_T(constraints, cost):
    have_res = False
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    try:
        prob.solve(verbose=False, solver=cvxpy.GUROBI, reoptimize=True)
        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            have_res = True
    except cvxpy.error.SolverError:
        pass
    return have_res


def weaving_constraints_and_costs(
        T, var_ev: cvxpy.Variable, veh: 'Game_A_Vehicle',
        var_other: cvxpy.Variable, other_veh: 'Game_A_Vehicle',
        ev_lc_cross_step, ot_lc_cross_step
):
    dt = veh.dt
    time_steps = np.arange(0, T + dt / 2, dt)
    A_xy = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])  # 纵向位置
    A_dxy = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps])  # 横向速度
    ev_x = A_xy @ var_ev[: 6]  # 纵向位置
    ev_vx = A_dxy @ var_ev[:6]
    ot_x = A_xy @ var_other[: 6]  # 纵向位置
    ot_vx = A_dxy @ var_other[:6]

    if ev_lc_cross_step < ot_lc_cross_step:  # EV先换道
        ev_x_interest = ev_x[ev_lc_cross_step: ot_lc_cross_step]
        ev_vx_interest = ev_vx[ev_lc_cross_step: ot_lc_cross_step]
        ot_x_interest = ot_x[ev_lc_cross_step: ot_lc_cross_step]
        ot_vx_interest = ot_vx[ev_lc_cross_step: ot_lc_cross_step]
    else:
        ev_x_interest = ev_x[ot_lc_cross_step: ev_lc_cross_step]
        ev_vx_interest = ev_vx[ot_lc_cross_step: ev_lc_cross_step]
        ot_x_interest = ot_x[ot_lc_cross_step: ev_lc_cross_step]
        ot_vx_interest = ot_vx[ot_lc_cross_step: ev_lc_cross_step]

    is_preceding = True if veh.x < other_veh.x else False
    ot_constraints, ot_cost = get_single_surr_constraints_and_costs(
        veh, ev_x_interest, ev_vx_interest, ot_x_interest, ot_vx_interest, other_veh.length, is_preceding
    )

    ev_constraints, ev_cost = get_single_surr_constraints_and_costs(
        other_veh, ot_x_interest, ot_vx_interest, ev_x_interest, ev_vx_interest, veh.length, not is_preceding
    )

    return ev_constraints, ot_constraints, ev_cost, ot_cost


def platoon_constraints_and_costs(
        T, var_ev: cvxpy.Variable, veh: 'Game_A_Vehicle',
        var_other: cvxpy.Variable, other_veh: 'Game_A_Vehicle',
        ev_lc_cross_step, ot_lc_cross_step
):
    dt = veh.dt
    time_steps = np.arange(0, T + dt / 2, dt)
    A_xy = np.array([[1, t, t ** 2, t ** 3, t ** 4, t ** 5] for t in time_steps])  # 纵向位置
    A_dxy = np.array([[0, 1, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4] for t in time_steps])  # 横向速度
    ev_x = A_xy @ var_ev[: 6]  # 纵向位置
    ev_vx = A_dxy @ var_ev[:6]
    ot_x = A_xy @ var_other[: 6]  # 纵向位置
    ot_vx = A_dxy @ var_other[:6]

    if ev_lc_cross_step < ot_lc_cross_step:  # EV先换道
        cross_before_step = ev_lc_cross_step
        cross_after_step = ot_lc_cross_step
    else:
        cross_before_step = ot_lc_cross_step
        cross_after_step = ev_lc_cross_step

    ev_x_before = ev_x[: cross_before_step]
    ev_vx_before = ev_vx[: cross_before_step]
    ot_x_before = ot_x[: cross_before_step]
    ot_vx_before = ot_vx[: cross_before_step]
    ev_x_after = ev_x[cross_after_step:]
    ev_vx_after = ev_vx[cross_after_step:]
    ot_x_after = ot_x[cross_after_step:]
    ot_vx_after = ot_vx[cross_after_step:]

    is_preceding = True if veh.x < other_veh.x else False
    ot_constraints_before, ot_cost_before = get_single_surr_constraints_and_costs(
        veh, ev_x_before, ev_vx_before, ot_x_before, ot_vx_before, other_veh.length, is_preceding
    )
    ev_constraints_before, ev_cost_before = get_single_surr_constraints_and_costs(
        other_veh, ot_x_before, ot_vx_before, ev_x_before, ev_vx_before, veh.length, not is_preceding
    )
    ot_constraints_after, ot_cost_after = get_single_surr_constraints_and_costs(
        veh, ev_x_after, ev_vx_after, ot_x_after, ot_vx_after, other_veh.length, is_preceding
    )
    ev_constraints_after, ev_cost_after = get_single_surr_constraints_and_costs(
        other_veh, ot_x_after, ot_vx_after, ev_x_after, ev_vx_after, veh.length, not is_preceding
    )

    ev_constraints = ev_constraints_before + ev_constraints_after
    ot_constraints = ot_constraints_before + ot_constraints_after
    ev_cost = cvxpy.max(cvxpy.hstack([ev_cost_before, ev_cost_after]))
    ot_cost = cvxpy.max(cvxpy.hstack([ot_cost_before, ot_cost_after]))

    return ev_constraints, ot_constraints, ev_cost, ot_cost


def opti_quintic_given_T_single(
        lc_surr: GameVehSurr, T,
        TP_traj=None, TR_traj=None, CP_traj=None, CR_traj=None,
        ori_lane: 'LaneAbstract' = None, target_lane: 'LaneAbstract' = None
) -> Optional[SolveRes]:
    EV: 'Game_A_Vehicle' = lc_surr.EV

    x_ev = cvxpy.Variable(12)  # fxt，fyt的五次多项式系数

    ev_constraints, ev_com_cost, ev_eff_cost, ev_route_cost = \
        get_traj_own_constraints_and_costs(x_ev, EV, target_lane, T)

    ev_constraints_surr, ev_safe_cost, ev_cross_step = get_surr_constraints_and_costs(
        x_ev, EV, T,
        lc_surr.TP, lc_surr.TR, lc_surr.CP, lc_surr.CR,
        TP_traj, TR_traj, CP_traj, CR_traj,
        target_lane, ori_lane
    )

    constraints = ev_constraints + ev_constraints_surr
    ev_total_cost = ((1 - EV.rho) * (ev_safe_cost + ev_com_cost) + EV.rho * ev_eff_cost + ev_route_cost)

    have_res = solve_quintic_given_T(constraints, ev_total_cost)
    if not have_res:
        return None

    ev_res = x_ev.value
    safe_cost_value = ev_safe_cost.value
    com_cost_value = ev_com_cost.value
    eff_cost_value = ev_eff_cost.value
    route_cost_value = ev_route_cost
    ev_total_cost = ev_total_cost.value
    times = np.arange(0, T + EV.dt / 2, EV.dt)
    ev_solve_res = SolveRes(
        ev_res, times, safe_cost_value, com_cost_value, eff_cost_value, route_cost_value, ev_total_cost
    )

    return ev_solve_res


def opti_quintic_given_T_weaving(
        lc_surr: GameVehSurr, T,
        TP_traj=None, TRR_traj=None, CP_traj=None, CR_traj=None,
        ori_lane: 'LaneAbstract' = None, target_lane: 'LaneAbstract' = None
) -> tuple[Optional[SolveRes], Optional[SolveRes]]:
    EV: 'Game_A_Vehicle' = lc_surr.EV
    TR: 'Game_A_Vehicle' = lc_surr.TR  # noqa

    x_ev = cvxpy.Variable(12)  # fxt，fyt的五次多项式系数
    x_tr = cvxpy.Variable(12)  # y的五次多项式系数

    ev_constraints, ev_com_cost, ev_eff_cost, ev_route_cost = \
        get_traj_own_constraints_and_costs(x_ev, EV, target_lane, T)
    tr_constraints, tr_com_cost, tr_eff_cost, tr_route_cost = \
        get_traj_own_constraints_and_costs(x_tr, TR, ori_lane, T)

    ev_constraints_surr, ev_safe_cost_surr, ev_cross_step = get_surr_constraints_and_costs(
        x_ev, EV, T,
        lc_surr.TP, lc_surr.TRR, lc_surr.CP, lc_surr.CR,
        TP_traj, TRR_traj, CP_traj, CR_traj,
        target_lane, ori_lane
    )
    tr_constraints_surr, tr_safe_cost_surr, tr_cross_step = get_surr_constraints_and_costs(
        x_tr, TR, T,
        lc_surr.CP, lc_surr.CR, lc_surr.TP, lc_surr.TRR,
        CP_traj, CR_traj, TP_traj, TRR_traj,
        ori_lane, target_lane
    )

    ev_ot_constraints, ot_ev_constraints, ev_ot_safe_cost, ot_ev_safe_cost = weaving_constraints_and_costs(
        T, x_ev, EV,
        x_tr, TR,
        ev_cross_step, tr_cross_step
    )

    ev_safe_cost = cvxpy.max(cvxpy.hstack([ev_safe_cost_surr, ev_ot_safe_cost]))
    tr_safe_cost = cvxpy.max(cvxpy.hstack([tr_safe_cost_surr, ot_ev_safe_cost]))

    ev_total_cost = ((1 - EV.rho) * (ev_safe_cost + ev_com_cost) + EV.rho * ev_eff_cost + ev_route_cost)
    tr_total_cost = ((1 - TR.rho) * (tr_safe_cost + tr_com_cost) + TR.rho * tr_eff_cost + tr_route_cost)

    total_cost = ev_total_cost + TR.game_co * tr_total_cost
    constraints = (
            ev_constraints + tr_constraints +
            ev_constraints_surr + tr_constraints_surr +
            ev_ot_constraints + ot_ev_constraints
    )

    have_res = solve_quintic_given_T(constraints, total_cost)
    if not have_res:
        return None, None

    times = np.arange(0, T + EV.dt / 2, EV.dt)

    ev_res = x_ev.value
    ev_safe_cost_value = ev_safe_cost.value
    ev_com_cost_value = ev_com_cost.value
    ev_eff_cost_value = ev_eff_cost.value
    ev_route_cost_value = ev_route_cost
    ev_total_cost = ev_total_cost.value

    ev_solve_res = SolveRes(
        ev_res, times, ev_safe_cost_value, ev_com_cost_value, ev_eff_cost_value, ev_route_cost_value, ev_total_cost
    )

    tr_res = x_tr.value
    tr_safe_cost_value = tr_safe_cost.value
    tr_com_cost_value = tr_com_cost.value
    tr_eff_cost_value = tr_eff_cost.value
    tr_route_cost_value = tr_route_cost
    tr_total_cost = tr_total_cost.value

    tr_solve_res = SolveRes(
        tr_res, times, tr_safe_cost_value, tr_com_cost_value, tr_eff_cost_value, tr_route_cost_value, tr_total_cost
    )

    return ev_solve_res, tr_solve_res


def opti_quintic_given_T_platoon(
        lc_surr: GameVehSurr, T,
        TP_traj=None, TR_traj=None, CP_traj=None, CRR_traj=None,
        ori_lane: 'LaneAbstract' = None, target_lane: 'LaneAbstract' = None
) -> tuple[Optional[SolveRes], Optional[SolveRes]]:
    EV: 'Game_A_Vehicle' = lc_surr.EV
    CR: 'Game_A_Vehicle' = lc_surr.CR  # noqa

    x_ev = cvxpy.Variable(12)  # fxt，fyt的五次多项式系数
    x_cr = cvxpy.Variable(12)  # y的五次多项式系数

    ev_constraints, ev_com_cost, ev_eff_cost, ev_route_cost = \
        get_traj_own_constraints_and_costs(x_ev, EV, target_lane, T)
    cr_constraints, cr_com_cost, cr_eff_cost, cr_route_cost = \
        get_traj_own_constraints_and_costs(x_cr, CR, target_lane, T)

    ev_constraints_surr, ev_safe_cost_surr, ev_cross_step = get_surr_constraints_and_costs(
        x_ev, EV, T,
        lc_surr.TP, lc_surr.TR, lc_surr.CP, lc_surr.CRR,
        TP_traj, TR_traj, CP_traj, CRR_traj,
        target_lane, ori_lane
    )
    cr_constraints_surr, cr_safe_cost_surr, cr_cross_step = get_surr_constraints_and_costs(
        x_cr, CR, T,
        lc_surr.TP, lc_surr.TR, lc_surr.CP, lc_surr.CRR,
        TP_traj, TR_traj, CP_traj, CRR_traj,
        target_lane, ori_lane
    )

    ev_ot_constraints, ot_ev_constraints, ev_ot_safe_cost, ot_ev_safe_cost = platoon_constraints_and_costs(
        T, x_ev, EV,
        x_cr, CR,
        ev_cross_step, cr_cross_step
    )

    ev_safe_cost = cvxpy.max(cvxpy.hstack([ev_safe_cost_surr, ev_ot_safe_cost]))
    cr_safe_cost = cvxpy.max(cvxpy.hstack([cr_safe_cost_surr, ot_ev_safe_cost]))

    ev_total_cost = ((1 - EV.rho) * (ev_safe_cost + ev_com_cost) + EV.rho * ev_eff_cost + ev_route_cost)
    cr_total_cost = ((1 - CR.rho) * (cr_safe_cost + cr_com_cost) + CR.rho * cr_eff_cost + cr_route_cost)

    total_cost = ev_total_cost + CR.game_co * cr_total_cost  # 等权
    constraints = (
            ev_constraints + cr_constraints +
            ev_constraints_surr + cr_constraints_surr +
            ev_ot_constraints + ot_ev_constraints
    )

    have_res = solve_quintic_given_T(constraints, total_cost)
    if not have_res:
        return None, None

    times = np.arange(0, T + EV.dt / 2, EV.dt)

    ev_res = x_ev.value
    ev_safe_cost_value = ev_safe_cost.value
    ev_com_cost_value = ev_com_cost.value
    ev_eff_cost_value = ev_eff_cost.value
    ev_route_cost_value = ev_route_cost
    ev_total_cost = ev_total_cost.value
    ev_solve_res = SolveRes(
        ev_res, times,
        ev_safe_cost_value, ev_com_cost_value, ev_eff_cost_value, ev_route_cost_value, ev_total_cost
    )

    cr_res = x_cr.value
    cr_safe_cost_value = cr_safe_cost.value
    cr_com_cost_value = cr_com_cost.value
    cr_eff_cost_value = cr_eff_cost.value
    cr_route_cost_value = cr_route_cost
    cr_total_cost = cr_total_cost.value
    cr_solve_res = SolveRes(
        cr_res, times,
        cr_safe_cost_value, cr_com_cost_value, cr_eff_cost_value, cr_route_cost_value, cr_total_cost
    )

    return ev_solve_res, cr_solve_res
