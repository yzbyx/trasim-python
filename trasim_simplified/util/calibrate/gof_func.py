# -*- coding: utf-8 -*-
# @Time : 2023/10/14 18:33
# @Author : yzbyx
# @File : gof_func.py
# Software: PyCharm
"""
# Focusing on spacing, regardless of the model and dataset, GoFs which are not based on percentage errors
# i.e., RMSE(s), Theil’s U (s) and MAE(s) are always more preferable than percentage-based GoFs i.e., RMSPE and MAPE
#
# About calibration of car-following dynamics of automated and human-driven vehicles: Methodology, guidelines and codes
"""
import numpy as np


def RMSE(sim_x, sim_v, obs_x, obs_v, obs_lx, eval_params=None):
    if eval_params is None:
        eval_params = ["dhw"]
    if "dhw" in eval_params:
        dhw_sim = np.array(sim_x) - np.array(obs_lx)
        dhw_obs = np.array(obs_x) - np.array(obs_lx)
        RMSE_dhw = np.sqrt(np.mean(np.power(dhw_sim - dhw_obs, 2)))
        return RMSE_dhw
    if "v" in eval_params:
        RMSE_v = np.sqrt(np.mean(np.power(sim_v - obs_v, 2)))
        return RMSE_v


def RMSPE(sim_x, sim_v, obs_x, obs_v, obs_lx, eval_params=None, alpha_x=1, alpha_v=1):
    if eval_params is None:
        eval_params = ["dhw"]
    RMSPE_dhw, RMSPE_v = 0, 0
    if "dhw" in eval_params:
        dhw_sim = np.array(sim_x) - np.array(obs_lx)
        dhw_obs = np.array(obs_x) - np.array(obs_lx)
        RMSPE_dhw = np.sqrt(np.mean(np.power(
            (dhw_sim - dhw_obs) / dhw_obs, 2)))
    if "v" in eval_params:
        # 极低速度不计算
        v_th = 1e-1
        temp = (sim_v - obs_v) / obs_v
        RMSPE_v = np.sqrt(np.mean(np.power(
            temp[np.where(np.array(obs_v) > v_th)], 2)))
    return (alpha_x * RMSPE_dhw + alpha_v * RMSPE_v) / len(eval_params)


def Theil_s_U(sim_x, sim_v, obs_x, obs_v, obs_lx, eval_params=None, alpha_x=1, alpha_v=1):
    if eval_params is None:
        eval_params = ["dhw"]
    U_dhw, U_v = 0, 0
    if "dhw" in eval_params:
        dhw_sim = np.array(sim_x) - np.array(obs_lx)
        dhw_obs = np.array(obs_x) - np.array(obs_lx)
        U_dhw = (RMSE(sim_x, sim_v, obs_x, obs_v, obs_lx, eval_params=["dhw"])
                 / (np.std(dhw_sim) + np.std(dhw_obs)))
    if "v" in eval_params:
        U_v = (RMSE(sim_x, sim_v, obs_x, obs_v, obs_lx, eval_params=["v"])
               / (np.std(sim_v) + np.std(obs_v)))
    return (alpha_x * U_dhw + alpha_v * U_v) / len(eval_params)
