# -*- coding: utf-8 -*-
# @Time : 2023/10/10 14:45
# @Author : yzbyx
# @File : lane.py
# Software: PyCharm
from typing import Optional, Union

import numpy as np


# TODO
class MatrixLaneCal:
    """
    单车道开边界、同跟驰模型的仿真矩阵计算
    """
    def __init__(self):
        self.car_id_array: Optional[np.ndarray] = None
        self.car_pos_array: Optional[np.ndarray] = None
        self.car_speed_array: Optional[np.ndarray] = None
        self.car_acc_array: Optional[np.ndarray] = None

    def init(self, car_num: int, car_length: float, following_model: str,
             car_init_pos: Union[list[float], np.ndarray],
             car_init_speed: Union[list[float], np.ndarray]):
        self.car_id_array = np.arange(car_num) + 1
        self.car_pos_array = np.array(car_init_pos)
        self.car_speed_array = np.array(car_init_speed)
        self.car_acc_array = np.zeros(car_num)
