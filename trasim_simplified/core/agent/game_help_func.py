# -*- coding: utf-8 -*-
# @Time : 2025/5/12 20:58
# @Author : yzbyx
# @File : game_help_func.py
# Software: PyCharm
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle


def get_y_constraint(veh: "Vehicle", lc_direction):
    if lc_direction == -1:
        y1 = veh.lane.left_neighbour_lane.y_center
        if veh.y < veh.lane.y_center:
            y_limit_low = veh.lane.y_right
        else:
            y_limit_low = veh.lane.y_center - veh.lane.width / 4
        target_lane = veh.lane.left_neighbour_lane
        y_limit_up = target_lane.y_center + target_lane.width / 4
        y_middle = target_lane.y_right
    elif lc_direction == 1:
        y1 = veh.lane.right_neighbour_lane.y_center
        if veh.y < veh.lane.y_center:
            y_limit_up = veh.lane.y_center + veh.lane.width / 4
        else:
            y_limit_up = veh.lane.y_left
        target_lane = veh.lane.right_neighbour_lane
        y_limit_low = target_lane.y_center - target_lane.width / 4
        y_middle = target_lane.y_left
    else:
        y1 = veh.lane.y_center
        if veh.y < y1 - veh.lane.width / 4:
            y_limit_low = veh.lane.y_right
            y_limit_up = veh.lane.y_center + veh.lane.width / 4
        elif veh.y > y1 + veh.lane.width / 4:
            y_limit_low = veh.lane.y_center - veh.lane.width / 4
            y_limit_up = veh.lane.y_left
        else:
            y_limit_low = veh.lane.y_center - veh.lane.width / 4
            y_limit_up = veh.lane.y_center + veh.lane.width / 4
        y_middle = veh.lane.y_center

    return y1, y_limit_low, y_limit_up, y_middle


def get_y_constraint_simple(veh: "Vehicle", lc_direction):
    if lc_direction == -1:
        y1 = veh.lane.left_neighbour_lane.y_center
        y_limit_low = veh.lane.y_right
        y_limit_up = veh.lane.left_neighbour_lane.y_left
        y_middle = veh.lane.left_neighbour_lane.y_right
    elif lc_direction == 1:
        y1 = veh.lane.right_neighbour_lane.y_center
        y_limit_low = veh.lane.right_neighbour_lane.y_right
        y_limit_up = veh.lane.y_left
        y_middle = veh.lane.right_neighbour_lane.y_left
    else:
        y1 = veh.lane.y_center
        y_limit_low = veh.lane.y_right
        y_limit_up = veh.lane.y_left
        y_middle = None

    return y1, y_limit_low, y_limit_up, y_middle


def cal_other_cost(veh, rho_hat, traj, other_traj_s,
                   v_length_s, route_cost=0, print_cost=False, stra_info=None):
    cost_lambda = veh.cal_cost_by_traj(
        traj, other_traj_s, v_length_s,
        return_lambda=True, route_cost=route_cost,
        print_cost=print_cost, stra_info=stra_info
    )
    cost_hat = cost_lambda(rho_hat)
    real_cost = cost_lambda(veh.rho)
    return cost_hat, real_cost, cost_lambda


def get_TR_real_stra(cost_df_temp, EV_stra, CR_stra, TP_stra, CP_stra, traj_data_opti, traj_data_full):
    # 找到最小的TR_cost值
    min_TR_cost_idx_real = cost_df_temp.groupby(
        by=['ego_stra', 'CR_stra', 'TP_stra', 'CP_stra']
    )['TR_cost_real'].idxmin()
    min_cost_idx_real = cost_df_temp.loc[min_TR_cost_idx_real]["total_cost"].idxmin()  # 找到最小的cost值
    TR_stra = cost_df_temp.loc[min_cost_idx_real]["TR_stra"]
    TR_cost = cost_df_temp.loc[min_cost_idx_real]["TR_cost_real"]

    TR_index = cost_df_temp[
        (cost_df_temp["ego_stra"] == EV_stra) & (cost_df_temp["TR_stra"] == TR_stra) &
        (cost_df_temp["CR_stra"] == CR_stra) & (cost_df_temp["TP_stra"] == TP_stra) &
        (cost_df_temp["CP_stra"] == CP_stra)
    ]["index"].values[0]

    traj_data_real = traj_data_full[TR_index]
    TR_esti_lambda = traj_data_opti.TR_cost_lambda
    TR_real_lambda = traj_data_real.TR_cost_lambda

    return TR_stra, TR_cost, TR_esti_lambda, TR_real_lambda


def get_CR_real_stra(cost_df_temp, EV_stra, TR_stra, TP_stra, CP_stra, traj_data_opti, traj_data_full):
    # 找到最小的CR_cost值
    min_CR_cost_idx_real = cost_df_temp.groupby(
        by=['ego_stra', 'TR_stra', 'TP_stra', 'CP_stra']
    )['CR_cost_real'].idxmin()
    min_cost_idx_real = cost_df_temp.loc[min_CR_cost_idx_real]["total_cost"].idxmin()  # 找到最小的cost值
    CR_stra = cost_df_temp.loc[min_cost_idx_real]["CR_stra"]
    CR_cost = cost_df_temp.loc[min_cost_idx_real]["CR_cost_real"]

    CR_index = cost_df_temp[
        (cost_df_temp["ego_stra"] == EV_stra) & (cost_df_temp["CR_stra"] == CR_stra) &
        (cost_df_temp["TP_stra"] == TP_stra) & (cost_df_temp["CP_stra"] == CP_stra) &
        (cost_df_temp["TR_stra"] == TR_stra)
    ]["index"].values[0]

    traj_data_real = traj_data_full[CR_index]
    CR_esti_lambda = traj_data_opti.CR_cost_lambda
    CR_real_lambda = traj_data_real.CR_cost_lambda

    return CR_stra, CR_cost, CR_esti_lambda, CR_real_lambda
