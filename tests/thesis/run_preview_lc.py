# -*- coding: utf-8 -*-
# @time : 2025/4/7 11:28
# @Author : yzbyx
# @File : run_preview_lc.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

from pyswarm import pso

from traj_process.util.plot_helper import get_fig_ax
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM, SECTION_TYPE, MARKING_TYPE, V_CLASS, RouteType
from trasim_simplified.core.frame.micro.open_lane import THW_DISTRI, LaneOpen
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.util.timer import timer_no_log


def get_road():
    road_length = 600
    lane_num = 2

    road = Road(road_length)
    lanes: list[LaneOpen] = road.add_lanes(lane_num, is_circle=False)
    for i in range(lane_num):
        if i != lane_num - 1:
            lanes[i].set_speed_limit(30)
        else:
            lanes[i].set_speed_limit(22.2)

        if i == lane_num - 2:
            lanes[i].set_marking_type(
                [
                    (MARKING_TYPE.SOLID, MARKING_TYPE.DASHED),
                ],
                [0, road_length],
            )
        if i == lane_num - 1:
            lanes[i].set_marking_type(
                [
                    (MARKING_TYPE.DASHED, MARKING_TYPE.SOLID),
                ],
                [0, road_length],
            )

    return road, lanes


@timer_no_log
def run_road(preview_time=3.0):
    _cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True, "g_tau": 3, "kdv": 0.3}
    _car_param = {}
    take_over_index = -1
    follower_index = -1
    v_length = 5.0
    dt = 0.1

    road, lanes = get_road()

    veh = lanes[1].car_insert(
        5, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        10, 10, 0,
        CFM.IDM, {}, {"color": COLOR.blue},
        lc_name=LCM.MOBIL, lc_param={}
    )

    point_list = []
    # 估计轨迹
    veh.target_lane = road.lane_list[0]
    veh.preview_time = preview_time

    point_list = []
    step_s = np.arange(round(20 / dt))
    time_s = step_s * dt
    for _ in step_s:
        point_list.append(veh.get_traj_point())
        delta = veh.cf_lateral_control()
        acc = 0
        veh.update_state(acc, delta)

    fig, ax = get_fig_ax()
    traj = [
        (point_list[i].x, point_list[i].y)
        for i in range(len(point_list))
    ]
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], color='b', label="traj")

    # 画车道
    for lane in lanes:
        ax.hlines(lane.y_center, 0, lane.lane_length, color='k', label=f"lane {lane.index}")

    plt.show()


def calculate_settling_time(times, deviations, convergence_threshold=0.1, start_time_offset=0.0):
    """
    计算信号收敛到指定阈值内的时长。

    参数:
    times (list or np.array): 时间戳列表。
    deviations (list or np.array): 对应时间戳的偏差信号 (已减去稳态目标值)。
    convergence_threshold (float): 收敛的阈值 (例如 0.1m)。
    start_time_offset (float): 响应开始的参考时间点，默认为0。

    返回:
    float: 收敛时长。如果从未收敛，可以返回一个标记值 (例如 float('inf') 或 None)。
    """

    # 从后向前查找，确保是“持续保持”
    # 找到最后一个超出阈值的点
    last_unsettled_index = -1
    for i in range(len(deviations) - 1, -1, -1):
        if abs(deviations[i]) > convergence_threshold:
            last_unsettled_index = i
            break

    if last_unsettled_index == -1:
        # 如果所有点都在阈值内 (从一开始就满足)
        # 或者如果系统响应是从一个扰动开始，我们关心从扰动开始的时间
        # 这里假设如果一开始就在阈值内，那么收敛时间是0（相对于扰动发生时刻）
        # 如果数据点本身就很少，或者开始就在阈值内，需要根据具体情况定义
        # 通常我们会假设系统是从一个非稳定状态开始的
        # 如果第一个点就在阈值内，且后面都保持，则收敛时间为第一个点的时间减去参考开始时间
        if abs(deviations[0]) <= convergence_threshold:
            return times[0] - start_time_offset  #  或者直接返回 times[0] 如果 start_time_offset 就是 times[0]
        else:  #这种情况不应该发生，因为 last_unsettled_index 会被更新
            return float('inf')

    # 如果所有点都超出阈值，则 last_unsettled_index 会是最后一个索引
    # 这种情况意味着从未稳定
    if last_unsettled_index == len(deviations) - 1:
        return float('inf')

        # 稳定时间是最后一个超出阈值点之后的一个点的时间
    settled_time_point = times[last_unsettled_index + 1]

    return settled_time_point - start_time_offset


def run_opti():
    road, lanes = get_road()
    dt = 0.1

    # --------------------------------------------------------------------------
    # 1. 定义你的PD控制器仿真和代价函数 (这是你需要重点修改的部分)
    # --------------------------------------------------------------------------
    def simulate_pd_controller(kp, kd, preview_time):
        """
        这是一个简化的PD控制器和被控对象的仿真占位符。
        你需要用你真实的系统模型和仿真逻辑替换这部分。

        参数:
        kp (float): 比例增益
        kd (float): 微分增益
        target_angle (float): 目标前轮转角 (例如，阶跃输入)
        duration (float): 仿真时长 (秒)
        dt (float): 仿真步长 (秒)

        返回:
        cost (float): 基于仿真结果的代价值 (越小越好)
        """
        veh = lanes[1].car_insert(
            5, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
            10, 10, 0,
            CFM.IDM, {}, {"color": COLOR.blue},
            lc_name=LCM.MOBIL, lc_param={}
        )

        dist_list = []
        # 估计轨迹
        veh.target_lane = road.lane_list[0]
        veh.preview_time = preview_time
        veh.K_P = kp
        veh.K_D = kd

        step_s = np.arange(round(20 / dt))
        time_s = step_s * dt
        for _ in step_s:
            dist_list.append(veh.y - veh.target_lane.y_center)
            delta = veh.cf_lateral_control()
            acc = 0
            veh.update_state(acc, delta)

        # 计算震荡峰值的收敛至0.1m的时长
        t = calculate_settling_time(time_s, dist_list, convergence_threshold=0.1, start_time_offset=0.0)

        if max(dist_list) < 0.5:
            return float('inf')  # 如果最大偏差超过0.1m，认为不收敛

        cost = max(dist_list) + t

        return cost

    # --------------------------------------------------------------------------
    # 2. 定义PSO的目标函数 (包裹上面的仿真函数)
    # --------------------------------------------------------------------------
    def objective_function(params):
        """
        PSO算法会调用这个函数。
        `params` 是一个包含 [kp, kd] 的数组。
        """
        preview_time = params[0]
        return simulate_pd_controller(1, 1, preview_time)

    # --------------------------------------------------------------------------
    # 3. 设置PSO参数和边界
    # --------------------------------------------------------------------------

    # 定义 Kp 和 Kd 的搜索范围 (下界和上界)
    # 这些边界需要根据你的系统和经验来设定，非常重要！
    # 例如：Kp 可能在 0.1 到 50 之间，Kd 可能在 0.01 到 10 之间
    # lb = [0.1, 0., 0.]  # Lower bounds for [Kp, Kd]
    # ub = [5.0, 2.0, 10.]  # Upper bounds for [Kp, Kd]
    lb = [0.]
    ub = [10.]

    # PSO 算法参数
    swarmsize = 50  # 粒子数量 (例如 20-100)
    maxiter = 100  # 最大迭代次数 (例如 50-200)
    minstep = 1e-8  # 迭代之间的最小步长 (收敛标准)
    minfunc = 1e-8  # 目标函数值的最小改变量 (收敛标准)
    omega = 0.5  # 惯性权重
    phip = 0.5  # 个体学习因子
    phig = 0.5  # 全局学习因子

    # print("开始PSO参数调优...")
    # print(f"参数搜索范围: Kp in [{lb[0]}, {ub[0]}], Kd in [{lb[1]}, {ub[1]}]")
    # print(f"粒子数: {swarmsize}, 最大迭代次数: {maxiter}")

    # --------------------------------------------------------------------------
    # 4. 运行PSO算法
    # --------------------------------------------------------------------------
    best_params, best_cost = pso(objective_function, lb, ub,
                                 swarmsize=swarmsize,
                                 maxiter=maxiter,
                                 minstep=minstep,
                                 minfunc=minfunc,
                                 omega=omega,
                                 phip=phip,
                                 phig=phig,
                                 debug=True)  # 设置 debug=True 可以看到每次迭代的详细信息

    # --------------------------------------------------------------------------
    # 5. 输出结果
    # --------------------------------------------------------------------------
    print("\nPSO调优完成!")
    print(f"最优参数 (Kp, Kd, T): {best_params}")
    print(f"对应的最小代价: {best_cost}")

    preview_time = best_params

    run_road(preview_time)


if __name__ == '__main__':
    run_road()
    # run_opti()
