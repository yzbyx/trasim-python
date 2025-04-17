# -*- coding: utf-8 -*-
# @Time : 2025/4/16 13:46
# @Author : yzbyx
# @File : scenario_loader.py
# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

from traj_process.processor.map_phrase.map_config import US_101_Config, ExpresswayA_Config
from trasim_simplified.core.agent import Vehicle, Game_H_Vehicle, Game_A_Vehicle
from trasim_simplified.core.constant import ScenarioTraj, V_TYPE, LCM, CFM, V_CLASS, COLOR, RouteType, ScenarioMode
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.util.scenario.util import make_road_from_osm


class Scenario:
    def __init__(self, traj_data: ScenarioTraj):
        self.weaving_offset = 20
        dataset_name = traj_data.dataset_name
        if dataset_name == "NGSIM":
            osm = r"E:\BaiduSyncdisk\datasets\NGSIM\NGSIM_US_101.osm"
            map_config = US_101_Config["weaving1"]
        else:
            osm = r"E:\BaiduSyncdisk\datasets\CitySim\CitySim_ExpresswayA.osm"
            map_config = ExpresswayA_Config["weaving2"]
        self.road: Road = make_road_from_osm(osm, map_config, self.weaving_offset)
        self.lanes = self.road.lane_list
        self.traj_data = traj_data
        self.max_step = 1e10
        self.ev_id = None
        self.surr_ids = {}

        self._load_vehicle()

    def _load_vehicle(self):
        for i, traj in enumerate([
            self.traj_data.EV_traj,
            self.traj_data.CP_traj, self.traj_data.CPP_traj,
            self.traj_data.TR_traj, self.traj_data.TRR_traj,
            self.traj_data.TP_traj, self.traj_data.TPP_traj,
            self.traj_data.OP_traj, self.traj_data.OPP_traj,
            self.traj_data.OR_traj, self.traj_data.ORR_traj,
        ]):
            if traj is None:
                continue
            self.max_step = min(self.max_step, len(traj))
            length = traj["length"].values[0]
            x = traj["myLocalLon"].values[0]
            y = traj["myLocalLat"].values[0]
            lane_id = traj["laneId"].values[0]
            heading = traj["heading"].values[0]
            speed = traj["speed"].values[0]
            if np.isnan(speed):
                speed = traj["speed"].values[1]
                assert not np.isnan(speed), f"speed is nan, {traj}"
            acc = traj["acceleration"].values[0]
            if np.isnan(acc):
                acc = traj["acceleration"].values[1]
                assert not np.isnan(acc), f"acceleration is nan, {traj}"
            vid = traj["trackId"].values[0]
            route_type = traj["routeClass"].values[0]

            if route_type == "mainline-mainline":
                route_type = RouteType.mainline
            elif route_type == "mainline-diverging":
                route_type = RouteType.diverge
            elif route_type == "merging-mainline":
                route_type = RouteType.merge
            elif route_type == "merging-diverging":
                route_type = RouteType.auxiliary
            else:
                raise ValueError(f"Unknown route type: {route_type}")

            if i == 0:
                self.ev_id = vid
                car = self._make_vehicle(lane_id, length, x, y, speed, acc, heading, vid, V_CLASS.GAME_AV, route_type)
            else:
                car = self._make_vehicle(lane_id, length, x, y, speed, acc, heading, vid, V_CLASS.GAME_HV, route_type)
            self.surr_ids.update({vid: (car, traj)})

    def _make_vehicle(self, lane_id, length, local_x, local_y, speed, acc, heading, vid,
                      car_class: V_CLASS, route_type: RouteType):
        local_x = local_x + self.road.start_weaving_pos + self.weaving_offset
        lane = self.lanes[lane_id]
        cf_name = CFM.KK if car_class != V_CLASS.GAME_AV else CFM.TPACC
        destination_lanes = self.road.mainline_end_indexes \
            if route_type in [RouteType.mainline, RouteType.merge] else self.road.auxiliary_end_indexes
        car = lane.car_insert(
            length, V_TYPE.PASSENGER, car_class,
            local_x, 0, 0,
            cf_name, {"v0": 30}, {"color": COLOR.red},
            lc_name=LCM.MOBIL, lc_param={}, destination_lanes=destination_lanes, route_type=route_type
        )
        car.ID = vid
        car.y = lane.y_center + local_y
        car.speed = speed
        car.acc = acc
        car.yaw = heading
        return car

    def set_vehicle_dynamic(self, car, lane, local_x, local_y, speed, acc, heading):
        local_x = local_x + self.road.start_weaving_pos + self.weaving_offset
        car.x = local_x
        car.y = lane.y_center + local_y
        car.speed = speed
        car.acc = acc
        car.yaw = heading
        return car

    def run(self, mode: ScenarioMode):
        if mode == ScenarioMode.NO_INTERACTION:
            for veh, _ in self.surr_ids.values():
                if veh.ID != self.ev_id:
                    veh.skip = True

        ev: Game_A_Vehicle = self.surr_ids[self.ev_id][0]
        ev.set_car_param({"color": COLOR.green})

        for step, stage in self.road.run(
                data_save=True, has_ui=True, frame_rate=-1, warm_up_step=0,
                sim_step=self.max_step, dt=0.1
        ):
            if stage == 4:
                # 显示ev真实轨迹
                traj = self.traj_data.EV_traj
                self.road.ui.ax.plot(traj["roadLon"].values,
                                     traj["roadLat"].values + self.road.lane_list[0].y_left,
                                     color='g', linewidth=1)
                self.road.ui.focus_on(ev)
                self.road.ui.plot_hist_traj()
                self.road.ui.plot_pred_traj()

                print("step:", step, ev)

                plt.pause(0.1)

            if stage == 4:
                if mode == ScenarioMode.NO_INTERACTION:
                    for vid, (car, traj) in self.surr_ids.items():
                        if vid != self.ev_id:
                            speed = traj["speed"].values[step]
                            x = traj["myLocalLon"].values[step]
                            y = traj["myLocalLat"].values[step]
                            lane_id = traj["laneId"].values[step]
                            heading = traj["heading"].values[step]
                            acc = traj["acceleration"].values[step]

                            self.set_vehicle_dynamic(car, self.lanes[lane_id], x, y, speed, acc, heading)
