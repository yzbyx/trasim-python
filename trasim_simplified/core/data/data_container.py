# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:20
# @Author : yzbyx
# @File : data_container.py
# @Software : PyCharm
from typing import TYPE_CHECKING, Optional

import pandas as pd

from trasim_simplified.core.constant import TrackInfo as Info


if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class DataContainer:
    def __init__(self, lane_abstract: 'LaneAbstract'):
        self.lane = lane_abstract
        self.data_pd: Optional[pd.DataFrame] = None
        self.save_info: set = set()
        self.total_car_list_has_data = None
        self.data_df: Optional[pd.DataFrame] = None

    def config(self, save_info=None, basic_info=True):
        """默认包含车辆ID"""
        if basic_info:
            save_info = [Info.lane_add_num, Info.step, Info.Time,
                         Info.id, Info.Preceding_ID, Info.x, Info.v, Info.a, Info.v_Length]
        if save_info is None:
            save_info = {}
        self.save_info.update(save_info)

    def add_basic_info(self):
        self.save_info.update([Info.lane_add_num, Info.id, Info.time, Info.step])

    def data_to_df(self):
        data: dict[str, list] = {info: [] for info in self.save_info}
        total_car_list_has_data = self.get_total_car_has_data()
        info_list = list(self.save_info)
        for info in info_list:
            for car in total_car_list_has_data:
                data[info].extend(car.get_data_list(info))
        self.data_df = pd.DataFrame(data, columns=info_list).sort_values(by=[Info.id, Info.time]).reset_index(drop=True)
        return self.data_df

    def get_total_car_has_data(self):
        """仿真完成后调用"""
        if self.total_car_list_has_data is None:
            car_on_lane_has_data = [car for car in self.lane.car_list if car.has_data()]
            self.total_car_list_has_data = car_on_lane_has_data + self.lane.out_car_has_data
        return self.total_car_list_has_data

    def get_data(self, id_, info_name):
        """仿真完成后调用"""
        return self.data_df[self.data_df[Info.id] == id_][info_name]


if __name__ == '__main__':
    print(Info)
