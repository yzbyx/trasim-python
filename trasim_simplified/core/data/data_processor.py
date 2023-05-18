# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:18
# @Author : yzbyx
# @File : data_processor.py
# @Software : PyCharm
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from trasim_simplified.core.data.data_container import Info as C_Info

if TYPE_CHECKING:
    from trasim_simplified.core.frame.lane_abstract import LaneAbstract


class DataProcessor:
    def __init__(self, frame_abstract: 'LaneAbstract'):
        self.frame = frame_abstract
        self.container = self.frame.data_container
        self.aggregate_all_result = {}
        """道路数据级别的集计结果"""
        self.aggregate_loop_result = {}
        """道路线圈检测器的集计结果"""
        self.aggregate_Edie_result = {}
        """Edie广义定义下的集计结果"""
        self.safe_result = {}
        """安全指标结果"""

        self.ttc_star = 1.5
        """判断TET的TTC阈值 [s]"""

        self.info: set = set()

    def config(self, cal_dict=None, add_all=True):
        if add_all:
            self.info.update(Info.get_all_info().values())
            return
        if cal_dict is None:
            cal_dict = {}
        self.info.update(cal_dict)

    def print_result(self):
        print("-" * 10 + "aggregate_all_result" + "-" * 10)
        for result in self.aggregate_all_result.items():
            print(result)
        print("-" * 10 + "aggregate_loop_result" + "-" * 10)
        for result in self.aggregate_loop_result.items():
            print(result)
        print("-" * 10 + "aggregate_Edie_result" + "-" * 10)
        for result in self.aggregate_Edie_result.items():
            print(result)
        print()

    @staticmethod
    def kqv_cal(df: pd.DataFrame, lane_length, lane_id=None, pos_range=None, time_range=None):
        assert df[C_Info.v] is not None, "调用本函数须使用record函数记录速度数据"
        if lane_id is not None:
            df = df[df[C_Info.lane_id] == lane_id]
        if pos_range is not None:
            df = df[(df[C_Info.x] >= pos_range[0]) & (df[C_Info.x] < pos_range[1])]
        if time_range is not None:
            df = df[((df[C_Info.time] >= time_range[0]) & (df[C_Info.time] < time_range[1]))]

        avg_speed = df[C_Info.v].mean()
        avg_k_by_car_num_lane_length = len(df[C_Info.id].unique()) / lane_length * 1000
        avg_q_by_v_k = avg_speed * 3.6 * avg_k_by_car_num_lane_length

        return {
            Info.avg_k_by_car_num_lane_length: avg_k_by_car_num_lane_length,
            Info.avg_q_by_v_k: avg_q_by_v_k,
            Info.avg_speed: avg_speed
        }

    def get_total_agg_info(self, name: str, func: Callable = np.average):
        total_car_list_has_data = self.container.get_total_car_has_data()
        temp = []
        if C_Info.a == name:
            for car in total_car_list_has_data:
                temp.extend(car.acc_list)
            return func(temp)
        elif C_Info.v == name:
            for car in total_car_list_has_data:
                temp.extend(car.speed_list)
            return func(temp)
        elif C_Info.gap == name:
            for car in total_car_list_has_data:
                temp.extend(car.gap_list)
            temp_array = np.array(temp)
            return func(temp_array[~ np.isnan(temp_array)])
        elif C_Info.dv == name:
            for car in total_car_list_has_data:
                temp.extend(car.dv_list)
            temp_array = np.array(temp)
            return func(temp_array[~ np.isnan(temp_array)])
        elif C_Info.dhw == name:
            for car in total_car_list_has_data:
                temp.extend(car.dhw_list)
            temp_array = np.array(temp)
            return func(temp_array[~ np.isnan(temp_array)])
        elif C_Info.thw == name:
            for car in total_car_list_has_data:
                temp.extend(car.thw_list)
            temp_array = np.array(temp)
            return func(temp_array[~ np.isnan(temp_array)])

    def aggregate(self):
        """集计指标计算"""
        # 由于总车辆数恒定，因此直接对所有车辆数据求平均是可行的
        avg_acc, avg_speed, avg_gap, avg_dv, avg_dhw, avg_thw = self.get_total_agg_info(C_Info.a),\
            self.get_total_agg_info(C_Info.v), self.get_total_agg_info(C_Info.gap), self.get_total_agg_info(C_Info.dv),\
            self.get_total_agg_info(C_Info.dhw), self.get_total_agg_info(C_Info.thw)
        # TODO：标准差这样做有待商榷
        std_acc, std_speed, std_gap, std_dv, std_dhw, std_thw = self.get_total_agg_info(C_Info.a, np.std), \
            self.get_total_agg_info(C_Info.v, np.std), self.get_total_agg_info(C_Info.gap, np.std),\
            self.get_total_agg_info(C_Info.dv, np.std), self.get_total_agg_info(C_Info.dhw, np.std),\
            self.get_total_agg_info(C_Info.thw, np.std)
        avg_q_by_thw = 3600 / avg_thw
        avg_k_by_dhw = 1000 / avg_dhw
        avg_v_q_div_k_by_thw_dhw = avg_q_by_thw / avg_k_by_dhw / 3.6

        harmonic_avg_speed = 1 / np.mean(1 / self.container.data_df[C_Info.v])

        avg_k_by_car_num_lane_length = sum(self.frame.car_num_list) / self.frame.lane_length * 1000
        avg_q_by_v_k = avg_speed * 3.6 * avg_k_by_car_num_lane_length

        self.aggregate_all_result.update({
            Info.avg_acc: avg_acc,
            Info.avg_speed: avg_speed,
            Info.harmonic_avg_speed: harmonic_avg_speed,
            Info.avg_gap: avg_gap,
            Info.avg_dv: avg_dv,
            Info.avg_dhw: avg_dhw,
            Info.avg_thw: avg_thw,
            Info.std_acc: std_acc,
            Info.std_speed: std_speed,
            Info.std_gap: std_gap,
            Info.std_dv: std_dv,
            Info.std_dhw: std_dhw,
            Info.std_thw: std_thw,
            Info.avg_q_by_thw: avg_q_by_thw,
            Info.avg_k_by_dhw: avg_k_by_dhw,
            Info.avg_v_q_div_k_by_thw_dhw: avg_v_q_div_k_by_thw_dhw,
            Info.avg_q_by_v_k: avg_q_by_v_k,
            Info.avg_k_by_car_num_lane_length: avg_k_by_car_num_lane_length})
        return self.aggregate_all_result

    def aggregate_as_detect_loop(self, pos, width, d_step: int, step_range: Sequence[int, int] = None):
        """
        以传感线圈的方式检测交通参数（HCM的平均速度定义）

        :param step_range: 总的检测始末仿真时刻(两个数)
        :param pos: 传感线圈起始位置 [m]
        :param width: 传感线圈宽度 [m]
        :param d_step: 每个检测周期的总仿真步
        :return: 包含顺序集计交通参数列表的字典
        """

        min_width = self.frame.dt * np.max(self.container.data_df[C_Info.v])
        assert width > min_width, f"至少将传感线圈的宽度设置在{min_width}以上！"
        is_return = False  # 传感器是否返回
        end_pos = pos + width
        if end_pos > self.frame.lane_length:
            is_return = True
            end_pos -= self.frame.lane_length
        if not is_return:
            pos_in = np.where((pos <= self.container.pos_data) & (self.container.pos_data <= end_pos))
        else:
            pos_in = np.where((pos <= self.container.pos_data) | (self.container.pos_data <= end_pos))
        if step_range is None:
            step_range = (self.frame.warm_up_step, self.frame.sim_step)
        time_sections = np.arange(step_range[0], step_range[1] + 1, d_step)
        time_sections_zip = zip(time_sections[:-1], time_sections[1:])

        # 计算时间平均车速
        # 每辆车的地点车速为线圈范围内速度点的平均（前提是数据点之间的采样时间间隔相同）
        time_avg_speed_list = []
        space_avg_speed_by_time_avg_speed_list = []
        q_list = []
        time_occ_list = []
        # HCM计算方式
        d_A_list = []
        t_A_list = []
        area_A = width * d_step * self.frame.dt

        for time_start, time_end in time_sections_zip:
            d_a = 0
            t_a = 0
            car_num_in = 0
            steps_has_car = 0
            avg_speed_list_per_dt = []
            for i in range(self.frame.car_num_list):
                target_pos = np.where(pos_in[1] == i)
                if len(target_pos[0]) == 0: continue

                # 提取某一辆车落在线圈范围内的数据行号；
                # pos_in[0]为所有落在检测器范围内数据点的行号，target_pos[0]为行号列表的某一辆车对应的索引号
                single_pos_all = np.sort(pos_in[0][target_pos[0]])
                # 数据行号需要加上预热仿真步，以便对应线圈检测的时间步；转换后的行号代表数据行对应的仿真总步数
                single_pos_all += self.frame.warm_up_step
                # 提取符合检测器检测时间段的数据行号索引，对应数据表的行号
                single_pos = single_pos_all[np.where((time_start <= single_pos_all) & (single_pos_all < time_end))]
                # 确定离去前的最后一个数据的数据表行号的索引
                last_pos_before_leave: list = np.where(np.diff(single_pos) != 1)[0].tolist()
                # 补充末端索引，确保完全覆盖数据表行号列表
                last_pos_before_leave.append(len(single_pos) - 1)
                start_pos = 0
                for end_pos in last_pos_before_leave:
                    single_data = self.container.speed_data[start_pos: end_pos + 1, i]
                    avg_speed_list_per_dt.append(np.mean(single_data))

                    # 对于只有一个数据点的情况，不纳入Edie广义定义中的时间和路程
                    if end_pos != start_pos:
                        pos_data = self.container.pos_data[start_pos: end_pos + 1, i]
                        #  如果轨迹点折返
                        if pos_data[0] > pos_data[-1]:
                            d_a += self.frame.lane_length - pos_data[0] + pos_data[-1]
                        else:
                            d_a += pos_data[-1] - pos_data[0]
                        # 这里不加1是因为轨迹选取是[start_pos, end_pos]
                        t_a += (end_pos - start_pos) * self.frame.dt

                    start_pos = end_pos + 1
                car_num_in += len(last_pos_before_leave)
                steps_has_car += len(single_pos)

            time_avg_speed_list.append(np.mean(avg_speed_list_per_dt))  # 时间平均车速
            # 地点车速的调和平均为空间平均车速
            space_avg_speed_by_time_avg_speed_list.append(np.mean(1 / np.array(avg_speed_list_per_dt)))
            q_list.append(car_num_in / (d_step * self.frame.dt))
            time_occ_list.append(steps_has_car / d_step)

            d_A_list.append(d_a)
            t_A_list.append(t_a)

        self.aggregate_loop_result["loop_vs(m/s)"] = time_avg_speed_list
        self.aggregate_loop_result["loop_vs_by_vt(m/s)"] = space_avg_speed_by_time_avg_speed_list
        self.aggregate_loop_result["loop_q(veh/h)"] = (np.array(q_list) * 3600).tolist()
        self.aggregate_loop_result["loop_time_occ"] = time_occ_list
        self.aggregate_loop_result["loop_k_by_time_occ(veh/km)"] = (np.array(time_occ_list) / width * 1000).tolist()

        self.aggregate_Edie_result["HCM_dA(m)"] = d_A_list
        self.aggregate_Edie_result["HCM_tA(s)"] = t_A_list
        self.aggregate_Edie_result["HCM_area_A(m*s)"] = area_A
        self.aggregate_Edie_result["HCM_qA(veh/h)"] = (np.array(d_A_list) / area_A * 3600).tolist()
        self.aggregate_Edie_result["HCM_kA(veh/km)"] = (np.array(t_A_list) / area_A * 1000).tolist()
        self.aggregate_Edie_result["HCM_vA(m/s)"] = (np.array(d_A_list) / np.array(t_A_list)).tolist()

        return [self.aggregate_loop_result, self.aggregate_Edie_result]

    @staticmethod
    def data_shear(temp_=None, pos_=None, time_=None, step_=None):
        return_index = list(np.where(np.diff(np.array(pos_)) < 0)[0])
        return_index_2 = list(np.where(np.diff(np.array(step_)) != 1)[0])
        return_index = list(set(return_index) | set(return_index_2))
        return_index.sort()

        return_index.insert(0, 0)
        if len(return_index) == 1:
            yield time_, temp_, [0, len(temp_)]
        else:
            for i in range(len(return_index)):
                if i == 0:
                    pos = (0, return_index[i + 1] + 1)
                elif i != len(return_index) - 1 and i != 0:
                    pos = (return_index[i] + 1, return_index[i + 1] + 1)
                else:
                    pos = (return_index[i] + 1, len(temp_))
                temp__ = temp_[pos[0]: pos[1]]
                time__ = time_[pos[0]: pos[1]]
                yield time__, temp__, pos


class Info:
    avg_acc = "avg_acc(m/s^2)"
    avg_speed = "avg_speed(m/s)"
    avg_gap = "avg_gap(m)"
    avg_dv = "avg_dv(m/s)"
    avg_dhw = "avg_dhw(m)"
    avg_thw = "avg_thw(s)"

    std_speed = "std_speed"
    std_acc = "std_acc"
    std_gap = "std_gap"
    std_dv = "std_dv"
    std_dhw = "std_dhw"
    std_thw = "std_thw"

    avg_q_by_thw = "avg_q(1/thw)(veh/h)"
    avg_k_by_dhw = "avg_k(1/dhw)(veh/km)"
    avg_v_q_div_k_by_thw_dhw = "avg_v_by_q(1/thw)/k(1/dhw)(m/s)"
    harmonic_avg_speed = "harmonic_avg_speed(m/s)"
    avg_k_by_car_num_lane_length = "avg_k(veh/km)"
    avg_q_by_v_k = "avg_q(v*k)(veh/h)"

    safe_ttc = "ttc(s)"
    safe_tet = "tet"
    safe_tit = "tit(s)"
    safe_picud = "picud(m)"

    @classmethod
    def get_all_info(cls):
        dict_ = Info.__dict__
        values = {}
        for key in dict_.keys():
            if isinstance(dict_[key], str) and key[:2] != "__":
                values.update({key: dict_[key]})
        return values

    @classmethod
    def get_safety_info(cls):
        dict_ = cls.get_all_info()
        values = {}
        for key in dict_.keys():
            if key[:4] == "safe":
                values.update({key: dict_[key]})
        return values


if __name__ == '__main__':
    print(Info.get_all_info())
