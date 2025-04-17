# -*- coding: utf-8 -*-
# @Time : 2025/4/16 13:57
# @Author : yzbyx
# @File : util.py
# Software: PyCharm
from traj_process.processor.map_phrase.lanelet_phrase import Lanelet
from trasim_simplified.core.constant import MARKING_TYPE, SECTION_TYPE
from trasim_simplified.core.frame.micro.road import Road


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
    upstream_length = sum([lanelet.lanelet_length[data[0]] for data in mainline_upstream_list])
    weaving_length = sum([lanelet.lanelet_length[data[0]] for data in mainline_weaving_list])

    upstream_end = upstream_length - weaving_offset
    downstream_start = upstream_length + weaving_length + weaving_offset

    road = Road(road_length)
    road.mainline_end_indexes = list(range(ramp_indexes[0]))
    road.auxiliary_end_indexes = ramp_indexes
    road.set_start_weaving_pos(upstream_end)
    road.set_end_weaving_pos(downstream_start)
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
