# -*- coding: utf-8 -*-
# @Time : 2025/4/16 13:57
# @Author : yzbyx
# @File : scenario_util.py
# Software: PyCharm
from typing import TYPE_CHECKING

import numpy as np

from traj_process.processor.map_phrase.lanelet_phrase import Lanelet
from trasim_simplified.core.agent.collision_risk import traj_data_TTC
from trasim_simplified.core.constant import MARKING_TYPE, SECTION_TYPE
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.core.constant import TrackInfo as C_Info

if TYPE_CHECKING:
    from trasim_simplified.util.scenario.scenario_loader import LcScenario


def make_road_from_osm(osm_file, map_config: dict, weaving_offset: int = 20):
    lanelet = Lanelet(osm_file)
    lanelet_in_lane = map_config['lanelet_in_lane']
    mainline_upstream_list = map_config['mainline_upstream_list']
    mainline_weaving_list = map_config['mainline_weaving_list']
    lanelet_base = map_config['lanelet_base']
    ramp_indexes = map_config['ramp_indexes']

    lane_num = len(lanelet_in_lane)
    road_length = sum([lanelet.lanelet_length[i] for i in lanelet_in_lane[0]])
    lane_width_s = [lanelet.get_lanelet(i).width for i in lanelet_base]
    upstream_length = sum([lanelet.lanelet_length[i] for i in mainline_upstream_list[0]])
    weaving_length = sum([lanelet.lanelet_length[i] for i in mainline_weaving_list[0]])

    upstream_end = upstream_length - weaving_offset
    downstream_start = upstream_length + weaving_length + weaving_offset

    road = Road(road_length)
    road.mainline_end_indexes = list(range(ramp_indexes[0]))
    road.auxiliary_end_indexes = ramp_indexes
    road.set_start_weaving_pos(upstream_end)
    road.set_end_weaving_pos(downstream_start)
    print("upstream_end:", upstream_end, "downstream_start:", downstream_start, "road_length:", road_length)
    lanes = road.add_lanes(lane_num, is_circle=False, lane_width_list=lane_width_s)
    for i in range(lane_num):
        if i == 0:
            if lane_num == 2:
                lanes[i].set_marking_type(
                    [
                        (MARKING_TYPE.SOLID, MARKING_TYPE.SOLID),
                        (MARKING_TYPE.SOLID, MARKING_TYPE.DASHED),
                        (MARKING_TYPE.SOLID, MARKING_TYPE.SOLID),
                    ],
                    [0, upstream_end, downstream_start, road_length],
                )
            else:
                lanes[i].set_marking_type(
                    [
                        (MARKING_TYPE.SOLID, MARKING_TYPE.DASHED),
                    ],
                    [0, road_length],
                )
        elif i == lane_num - 1:
            lanes[i].set_section_type(
                [
                    SECTION_TYPE.ON_RAMP, SECTION_TYPE.AUXILIARY, SECTION_TYPE.OFF_RAMP,
                ],
                [0, upstream_end, downstream_start, road_length],
            )
            lanes[i].set_marking_type(
                [
                    (MARKING_TYPE.SOLID, MARKING_TYPE.SOLID),
                    (MARKING_TYPE.DASHED, MARKING_TYPE.SOLID),
                    (MARKING_TYPE.SOLID, MARKING_TYPE.SOLID),
                ],
                [0, upstream_end, downstream_start, road_length],
            )
        elif i == lane_num - 2:
            lanes[i].set_marking_type(
                [
                    (MARKING_TYPE.DASHED, MARKING_TYPE.SOLID),
                    (MARKING_TYPE.DASHED, MARKING_TYPE.DASHED),
                    (MARKING_TYPE.DASHED, MARKING_TYPE.SOLID),
                ],
                [0, upstream_end, downstream_start, road_length],
            )
        else:
            lanes[i].set_marking_type(
                [
                    (MARKING_TYPE.DASHED, MARKING_TYPE.DASHED),
                ],
                [0, road_length],
            )
        road.set_start_weaving_pos(upstream_end)
        road.set_end_weaving_pos(downstream_start)
    return road


def cal_dist_indicator(traj):
    # 行驶距离偏差
    x_center = traj[C_Info.xCenterGlobal].to_numpy()
    y_center = traj[C_Info.yCenterGlobal].to_numpy()
    x_center_ori = traj[C_Info.xCenterGlobal + "_ori"].to_numpy()
    y_center_ori = traj[C_Info.yCenterGlobal + "_ori"].to_numpy()
    nan_indexes = np.isnan(x_center) | np.isnan(y_center) | np.isnan(x_center_ori) | np.isnan(y_center_ori)
    x_center = x_center[~nan_indexes]
    y_center = y_center[~nan_indexes]
    x_center_ori = x_center_ori[~nan_indexes]
    y_center_ori = y_center_ori[~nan_indexes]
    distance = np.linalg.norm(
        np.array([x_center - x_center_ori, y_center - y_center_ori]), axis=0
    )
    ade = np.mean(distance)
    fde = np.max(distance)

    # 计算车道变更
    lane_end = traj[C_Info.laneId].values[-1]
    lane_end_ori = traj[C_Info.laneId + "_ori"].values[-2]  # 仿真会多记一次
    lane_correct = lane_end == lane_end_ori

    return ade, fde, lane_correct


def relation_2_id_car_params(
        sce: 'LcScenario',
        car_params: dict[str, dict[str, float]]
):
    """
    场景中车辆的参数与关系
    :param sce: 场景
    :param car_params: 车辆参数{“车辆关系（TR、EV...）”: {"车辆参数": 数值}}
    :return:
    """
    car_params_s = {}
    for vid, (car, traj, name) in sce.surr_ids.items():
        if name in car_params:
            car_params_s.update({vid: car_params[name]})
    return car_params_s
