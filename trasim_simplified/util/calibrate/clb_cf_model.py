# -*- coding = uft-8 -*-
# @Time : 2022-04-06 23:24
# @Author : yzbyx
# @File : clb_cf_model.py
# @Software : PyCharm
import time

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from trasim_simplified.core.constant import CFM, TrackInfo as TI, Prefix
from trasim_simplified.core.kinematics.cfm import get_cf_default_param, get_cf_func

from trasim_simplified.util.calibrate.gof_func import RMSE
from trasim_simplified.util.calibrate.follow_sim import simulation, customize_sim

try:
    import geatpy as ea

    print("Using geatpy")
    __opti_package = "geatpy"
except ImportError as e:
    print(e)
    from sko.GA import GA

    ea = None
    print("Using sko")

    __opti_package = "sko"

cf_param_ranges = {
    CFM.IDM: {
        "s0": [0, 10], "v0": [10, 50], "T": [0, 10], "omega": [0.1, 10], "d": [0.1, 10],
        "delta": [1, 10]
    },
    CFM.GIPPS: {
        "a": [0, 10], "b": [-10, 0], "v0": [0, 40], "tau": [0, 2], "s": [0, 10], "b_hat": [-10, 0],
    },
    CFM.NON_LINEAR_GHR: {
        "m": [-5, 5], "l": [-5, 5], "a": [-10, 10], "tau": [0, 2]
    },
    CFM.WIEDEMANN_99: {
        "CC0": [0, 5], "CC1": [0, 5], "CC2": [0, 10], "CC3": [-20, 0], "CC4": [-5, 0], "CC5": [0, 5],
        "CC6": [0, 10], "CC7": [0, 5], "CC8": [0, 10], "CC9": [0, 10], "vDesire": [0, 40]
    },
    CFM.OPTIMAL_VELOCITY: {
        "a": [0, 10], "V0": [0, 40], "m": [0, 1], "bf": [0, 50], "bc": [0, 10]
    },
    CFM.ACC: {
        "k1": [0, 1], "k2": [0, 1], "thw": [0, 10], "s0": [0, 10]
    }
}
cf_param_types = {
    CFM.IDM: {"s0": 0, "v0": 0, "T": 0, "omega": 0, "d": 0, "delta": 1},
    CFM.GIPPS: {"a": 0, "b": 0, "v0": 0, "tau": 0, "s": 0, "b_hat": 0},
    CFM.NON_LINEAR_GHR: {"m": 0, "l": 0, "a": 0, "tau": 0},
    CFM.WIEDEMANN_99: {"CC0": 0, "CC1": 0, "CC2": 0, "CC3": 0, "CC4": 0, "CC5": 0, "CC6": 0, "CC7": 0, "CC8": 0,
                       "CC9": 0, "vDesire": 0},
    CFM.OPTIMAL_VELOCITY: {"a": 0, "V0": 0, "m": 0, "bf": 0, "bc": 0},
    CFM.ACC: {"k1": 0, "k2": 0, "thw": 0, "s0": 0}
}
cf_param_ins = {
    CFM.IDM: {
        "s0": [1, 1], "v0": [1, 1], "T": [1, 1], "omega": [1, 1], "d": [1, 1],
        "delta": [1, 1]
    },
    CFM.GIPPS: {
        "a": [1, 1], "b": [1, 1], "v0": [1, 1], "tau": [1, 1], "s": [1, 1], "b_hat": [1, 1],
    },
    CFM.NON_LINEAR_GHR: {
        "m": [1, 1], "l": [1, 1], "a": [1, 1], "tau": [1, 1]
    },
    CFM.WIEDEMANN_99: {
        "CC0": [1, 1], "CC1": [1, 1], "CC2": [1, 1], "CC3": [1, 1], "CC4": [1, 1], "CC5": [1, 1],
        "CC6": [1, 1], "CC7": [1, 1], "CC8": [1, 1], "CC9": [1, 1], "vDesire": [1, 1]
    },
    CFM.OPTIMAL_VELOCITY: {
        "a": [1, 1], "V0": [1, 1], "m": [1, 1], "bf": [1, 1], "bc": [1, 1]
    },
    CFM.ACC: {
        "k1": [1, 1], "k2": [1, 1], "thw": [1, 1], "s0": [1, 1]
    }
}


def ga_cal(cf_func, obs_x, obs_v, obs_lx, obs_lv, leaderL, dt, ranges: dict, ins: dict, types, seed, drawing=0):
    """
    :param cf_func: 跟驰模型函数
    :param dt: 仿真步长
    :param obs_x: 观测轨迹x
    :param obs_v: 观测轨迹v
    :param obs_lx: 观测轨迹leader x
    :param obs_lv: 观测轨迹leader v
    :param leaderL: 观测轨迹leaderL
    :param ranges: 参数范围{"a": 1, ...}
    :param ins: 参数边界是否包含{"a": [1, 1], ...}, 1表示包含，0表示不包含
    :param types: 参数类型，0：实数；1：整数
    :param seed: GA随机种子
    :param drawing: 是否绘图 0表示不绘图； 1表示绘制最终结果图； 2表示实时绘制目标空间动态图； 3表示实时绘制决策空间动态图。
    """
    param_names = list(ranges.keys())
    init_x = obs_x[0]
    init_v = obs_v[0]

    if __opti_package == "sko":
        def eval_vars(params):  # 定义目标函数（含约束）
            param = {k: v for k, v in zip(ranges.keys(), params)}
            x, v, _, _ = simulation(
                cf_func=cf_func, init_x=init_x, init_v=init_v, obs_lx=obs_lx, obs_lv=obs_lv,
                cf_param=param, leaderL=leaderL, dt=dt, update_method="Euler")
            return RMSE(sim_x=x, sim_v=v, obs_x=obs_x, obs_v=obs_v, obs_lx=obs_lx, eval_params=["dhw"])

        time_start = time.time()
        var_types = np.array([types[name] for name in param_names])
        ga = GA(func=eval_vars, n_dim=len(ranges), size_pop=50, max_iter=500, prob_mut=0.1,
                lb=[ranges[name][0] for name in param_names], ub=[ranges[name][1] for name in param_names],
                precision=np.where(var_types < 0.5, 1e-2, var_types), early_stop=50)
        best_x, best_y = ga.run()
        vars = {k: v for k, v in zip(ranges.keys(), best_x)}
        print(f"\n{cf_func.__name__}-seed{seed}:\nbest_y: {best_y}\nbest_x: {vars}\n"
              f"cal_time: {time.time() - time_start}s\n")
        res = {"ObjV": best_y, "Vars": {k: v for k, v in zip(ranges.keys(), best_x)}}

    elif __opti_package == "geatpy":
        @ea.Problem.single
        def eval_vars(params):  # 定义目标函数（含约束）
            param = {k: v for k, v in zip(ranges.keys(), params)}
            x, v, _, _ = simulation(
                cf_func=cf_func, init_x=init_x, init_v=init_v, obs_lx=obs_lx, obs_lv=obs_lv,
                cf_param=param, leaderL=leaderL, dt=dt, update_method="Euler")
            return RMSE(sim_x=x, sim_v=v, obs_x=obs_x, obs_v=obs_v, obs_lx=obs_lx, eval_params=["dhw"])

        problem = ea.Problem(name='test',
                             M=1,  # 目标维数
                             maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                             Dim=len(ranges),  # 决策变量维数
                             varTypes=[types[name] for name in param_names],  # 决策变量的类型列表，0：实数；1：整数
                             lb=[ranges[name][0] for name in param_names],  # 决策变量下界
                             ub=[ranges[name][1] for name in param_names],  # 决策变量上界
                             lbin=[ins[name][0] for name in param_names],  # 决策变量下边界
                             ubin=[ins[name][1] for name in param_names],  # 决策变量上边界
                             evalVars=eval_vars)
        algorithm = ea.soea_SEGA_templet(problem,
                                         ea.Population(Encoding='RI', NIND=min(max(10 * len(ranges), 40), 100)),
                                         MAXGEN=100 * len(ranges),  # 最大进化代数。
                                         logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                         trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                         maxTrappedCount=50)  # 进化停滞计数器最大上限值。
        algorithm.mutOper.Pm = 0.5  # 变异概率
        algorithm.recOper.XOVR = 0.7  # 重组概率

        # 求解
        res = ea.optimize(algorithm, verbose=False, seed=seed, drawing=drawing, outputMsg=False, drawLog=False,
                          saveFlag=False)
        print(f"{cf_func.__name__}-seed{seed}: {res['ObjV'][0]}, {res['Vars']}, {res['executeTime']}")
        res["Vars"] = {k: v for k, v in zip(ranges.keys(), res["Vars"][0])}
        res["ObjV"] = res["ObjV"][0]
    else:
        raise ValueError(f"Unknown optimization package: {__opti_package}")

    return res


def clb_run(cf_func, cf_name, obs_x_s, obs_v_s, obs_lx_s, obs_lv_s, leaderL_s, dt, seed,
            drawing=0, n_jobs=-1, parallel=True) -> list[dict]:
    """
    :param cf_func 跟驰模型加速度函数
    :param cf_name 跟驰模型名称
    :param obs_x_s 观测轨迹x
    :param obs_v_s 观测轨迹v
    :param obs_lx_s 观测轨迹leader x
    :param obs_lv_s 观测轨迹leader v
    :param leaderL_s 观测轨迹leaderL
    :param dt 仿真步长
    :param seed GA算法种子
    :param drawing GA计算过程中是否绘图
    :param n_jobs 并行计算的进程数，-1为全部进程
    :param parallel 是否不使用并行计算
    """
    if parallel:
        result = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(ga_cal)(cf_func=cf_func,
                                   obs_x=np.array(obs_x), obs_v=np.array(obs_v), obs_lx=np.array(obs_lx),
                                   obs_lv=np.array(obs_lv), leaderL=leaderL,
                                   dt=dt, ranges=cf_param_ranges[cf_name],
                                   types=cf_param_types[cf_name],
                                   ins=cf_param_ins[cf_name],
                                   seed=seed, drawing=drawing)
            for obs_x, obs_v, obs_lx, obs_lv, leaderL in zip(obs_x_s, obs_v_s, obs_lx_s, obs_lv_s, leaderL_s))
    else:
        result = []
        for obs_x, obs_v, obs_lx, obs_lv, leaderL in zip(obs_x_s, obs_v_s, obs_lx_s, obs_lv_s, leaderL_s):
            result.append(ga_cal(cf_func=cf_func,
                                 obs_x=np.array(obs_x), obs_v=np.array(obs_v), obs_lx=np.array(obs_lx),
                                 obs_lv=np.array(obs_lv), leaderL=leaderL,
                                 dt=dt, ranges=cf_param_ranges[cf_name],
                                 types=cf_param_types[cf_name],
                                 ins=cf_param_ins[cf_name],
                                 seed=seed, drawing=drawing))
    return result


def aggregate_result(result):
    avg_obj = np.mean([res['ObjV'] for res in result])
    avg_param = np.mean(np.array([list(res['Vars'].values()) for res in result]), axis=0)
    std_obj = np.std([res['ObjV'] for res in result])
    std_param = np.std(np.array([list(res['Vars'].values()) for res in result]), axis=0)
    return avg_obj, avg_param, std_obj, std_param


def get_test_sim_traj(cf_name, cf_func, dt, v_length=5):
    """
    按照从头车到后车的顺序返回状态值
    """
    param_ord = list(cf_param_ranges[cf_name].keys())
    cf_default_param = {name: get_cf_default_param(cf_name)[name] for name in param_ord}
    print(f"ori params: {cf_default_param}")

    # 生成轨迹
    x_lists, v_lists, a_lists, cf_a_lists = (
        customize_sim(leader_schedule=[(0, 300), (-1, 200), (0, 100), (1, 200), (0, 300)],
                      initial_states=[(0, 20, 0), (10, 20, 0)],
                      length_s=[v_length] * 2, cf_funcs=[cf_func], cf_params=[cf_default_param],
                      dt=dt))
    rmse = RMSE(sim_x=x_lists[1], sim_v=v_lists[1],
                obs_x=x_lists[1], obs_v=v_lists[1], obs_lx=v_length, eval_params=['dhw'])
    print(f"ori RMSE: {rmse}")
    return x_lists, v_lists, a_lists, cf_a_lists


def clb_param_to_df(id_s, clb_run_res: list[dict[str, dict]], cf_name):
    """
    将标定后的参数转换为DataFrame
    """
    vars_list = [[res["Vars"][name] for name in cf_param_ranges[cf_name].keys()] for res in clb_run_res]
    vars_array = np.array(vars_list)
    df = pd.DataFrame(vars_array, columns=list(cf_param_ranges[cf_name].keys()))
    df[TI.Pair_ID] = np.array(id_s)
    df["ObjV"] = np.array([res["ObjV"] for res in clb_run_res])
    return df


def get_clb_traj(df_follow_pair: dict[str | int, pd.DataFrame], cut_pos: dict[str | int, int],
                 clb_param_df: pd.DataFrame,
                 cf_func, cf_name, dt) -> dict[str | int, dict[str, pd.DataFrame]]:
    """
    使用标定后的跟驰模型参数，将后车轨迹转换为follow_pair的形式
    """
    final_data = {"dec": {}, "acc": {}}
    for pair_ID in df_follow_pair.keys():
        target = df_follow_pair[pair_ID].copy()
        clb_target = clb_param_df[clb_param_df[TI.Pair_ID] == pair_ID]
        clb_param = {name: clb_target[name].iloc[0] for name in cf_param_ranges[cf_name].keys()}
        sim_pos, sim_speed, sim_acc, sim_cf_acc = simulation(cf_func,
                                                             init_v=target[TI.v].iloc[0], init_x=target[TI.x].iloc[0],
                                                             obs_lx=target[Prefix.leader + TI.x],
                                                             obs_lv=target[Prefix.leader + TI.v],
                                                             cf_param=clb_param, dt=dt,
                                                             leaderL=target[Prefix.leader + TI.v_Length].unique()[0])
        target[TI.x] = sim_pos
        target[TI.v] = sim_speed
        target[TI.a] = sim_acc
        final_data["dec"].update({pair_ID: target.iloc[:cut_pos[pair_ID]]})
        final_data["acc"].update({pair_ID: target.iloc[cut_pos[pair_ID]:]})
    return final_data


def show_traj(cf_name, cf_param, dt, obs_x, obs_v, obs_lx, obs_lv, leaderL, traj_step=None, pair_ID=None):
    cf_func = get_cf_func(cf_name)
    cf_param = {k: v for k, v in zip(cf_param_ranges[cf_name].keys(), cf_param)}
    # 标定后跟驰模型轨迹仿真
    sim_pos, sim_speed, sim_acc, sim_cf_acc = simulation(cf_func, init_v=np.array(obs_v)[0],
                                                         init_x=np.array(obs_x)[0],
                                                         obs_lx=obs_lx, obs_lv=obs_lv, dt=dt,
                                                         cf_param=cf_param,
                                                         leaderL=leaderL)

    # 任选一条轨迹进行对比
    if traj_step is None:
        traj_step = range(len(obs_x))
    plt.plot(traj_step, obs_lx, label="obs_lx")
    plt.plot(traj_step, obs_x, label=f"obs_x: {pair_ID}")
    plt.plot(traj_step, sim_pos, label=f"sim_x: {pair_ID}")
    plt.legend()
    plt.show()
