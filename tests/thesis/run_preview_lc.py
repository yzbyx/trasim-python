# -*- coding: utf-8 -*-
# @time : 2025/4/7 11:28
# @Author : yzbyx
# @File : run_preview_lc.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

from traj_process.util.plot_helper import get_fig_ax
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM, SECTION_TYPE, MARKING_TYPE, V_CLASS, RouteType
from trasim_simplified.core.frame.micro.open_lane import THW_DISTRI, LaneOpen
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.util.timer import timer_no_log


@timer_no_log
def run_road():
    _cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True, "g_tau": 3, "kdv": 0.3}
    _car_param = {}
    take_over_index = -1
    follower_index = -1
    dt = 1
    warm_up_step = 0
    sim_step = warm_up_step + int(3600 / dt)
    offset_step = int(1800 / dt)

    road_length = 600
    lane_num = 2
    v_length = 5

    sim = Road(road_length)
    lanes: list[LaneOpen] = sim.add_lanes(lane_num, is_circle=False)
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

    veh = lanes[1].car_insert(
        v_length, V_TYPE.PASSENGER, V_CLASS.GAME_HV,
        10, 10, 0,
        CFM.IDM, _cf_param, {"color": COLOR.blue},
        lc_name=LCM.MOBIL, lc_param={}
    )

    traj_list = []
    # 估计轨迹
    veh.target_lane = sim.lane_list[0]
    veh.lane_changing = True
    veh.PREVIEW_TIME = 1
    for step in range(round(100 / dt)):
        traj_list.append(veh.get_traj_point())
        delta = veh.cf_lateral_control()
        acc = 0
        veh.update_state(acc, delta)

    fig, ax = get_fig_ax()
    traj = [
        (traj_list[i].x, traj_list[i].y)
        for i in range(len(traj_list))
    ]
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], color='b', label="traj")

    # 画车道
    for lane in lanes:
        ax.hlines(lane.y_center, 0, lane.lane_length, color='k', label=f"lane {lane.index}")

    plt.show()


if __name__ == '__main__':
    run_road()
