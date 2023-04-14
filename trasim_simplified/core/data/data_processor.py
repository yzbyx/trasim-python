# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:18
# @Author : yzbyx
# @File : data_processor.py
# @Software : PyCharm
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from trasim_simplified.core.frame.frame_abstract import FrameAbstract


class DataProcessor:
    def __init__(self, frame_abstract: 'FrameAbstract'):
        self.df: Optional[pd.DataFrame] = None
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

    def check_container(self):
        assert self.container.speed_data is not None and self.container.gap_data is not None \
               and self.container.dhw_data is not None and self.container.thw_data is not None \
               and self.container.dv_data is not None and self.container.pos_data is not None, \
            "调用本函数须使用record函数记录数据"

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

    def kqv_cal(self):
        assert self.container.speed_data is not None, "调用本函数须使用record函数记录速度数据"

        avg_speed = np.mean(self.container.speed_data)
        avg_k_by_car_num_lane_length = self.frame.car_num / self.frame.lane_length * 1000
        avg_q_by_v_k = avg_speed * 3.6 * avg_k_by_car_num_lane_length

        self.aggregate_all_result.update({
            Info.avg_speed: avg_speed,
            Info.avg_q_by_v_k: avg_q_by_v_k,
            Info.avg_k_by_car_num_lane_length: avg_k_by_car_num_lane_length
        })

        return self.aggregate_all_result

    def aggregate(self):
        self.check_container()
        # 由于总车辆数恒定，因此直接对所有车辆数据求平均是可行的
        avg_acc, avg_speed, avg_gap, avg_dv, avg_dhw, avg_thw = \
            (np.mean(data) for data in [self.container.acc_data, self.container.speed_data, self.container.gap_data,
                                        self.container.dv_data,
                                        self.container.dhw_data, self.container.thw_data])
        # TODO：标准差这样做有待商榷
        std_acc, std_speed, std_gap, std_dv, std_dhw, std_thw = \
            (np.std(data) for data in [self.container.acc_data, self.container.speed_data, self.container.gap_data,
                                       self.container.dv_data,
                                       self.container.dhw_data, self.container.thw_data])
        avg_q_by_thw = 3600 / avg_thw
        avg_k_by_dhw = 1000 / avg_dhw
        avg_v_q_div_k_by_thw_dhw = avg_q_by_thw / avg_k_by_dhw / 3.6

        harmonic_avg_speed = 1 / np.mean(1 / self.container.speed_data)

        avg_k_by_car_num_lane_length = self.frame.car_num / self.frame.lane_length * 1000
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

    def aggregate_as_detect_loop(self, pos, width, d_step: int, step_range: Sequence[int, int]=None):
        """
        以传感线圈的方式检测交通参数（HCM的平均速度定义）

        :param step_range: 总的检测始末仿真时刻(两个数)
        :param pos: 传感线圈起始位置 [m]
        :param width: 传感线圈宽度 [m]
        :param d_step: 每个检测周期的总仿真步
        :return: 包含顺序集计交通参数列表的字典
        """
        self.check_container()

        min_width = self.frame.dt * np.max(self.container.speed_data)
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
            for i in range(self.frame.car_num):
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
                car_num_in  += len(last_pos_before_leave)
                steps_has_car += len(single_pos)

            time_avg_speed_list.append(np.mean(avg_speed_list_per_dt))  # 时间平均车速
            space_avg_speed_by_time_avg_speed_list.append(np.mean(1 / np.array(avg_speed_list_per_dt)))  # 地点车速的调和平均为空间平均车速
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

    def cal_safety(self):
        if Info.safe_ttc in self.info:
            self.safe_result[Info.safe_ttc] = self.container.dhw_data / (- self.container.dv_data)
        if Info.safe_tet in self.info and Info.safe_ttc in self.info:
            where_ = np.where((self.safe_result[Info.safe_ttc] > 0) &
                              (self.safe_result[Info.safe_ttc] < self.ttc_star))
            result = np.zeros(self.container.speed_data.shape)
            result[where_] = 1
            self.safe_result[Info.safe_tet] = result
        if Info.safe_tit in self.info and Info.safe_ttc in self.info:
            where_ = np.where((self.safe_result[Info.safe_ttc] > 0) &
                              (self.safe_result[Info.safe_ttc] < self.ttc_star))
            result = np.zeros(self.container.speed_data.shape)
            result[where_] = self.ttc_star - self.safe_result[Info.safe_ttc][where_]
            self.safe_result[Info.safe_tit] = result
        if Info.safe_picud in self.info:
            d = 3.
            if "d" in self.frame.cf_model.get_param_map().keys():
                d = self.frame.cf_model.get_param_map()["d"]
            dec_dist = np.power(self.container.speed_data, 2) / (2 * d)
            dec_dist = np.concatenate([dec_dist, dec_dist[:, 0].reshape((-1, 1))], axis=1)
            react_dist = self.container.speed_data * self.frame.dt
            dist = self.container.gap_data
            self.safe_result[Info.safe_picud] = dist + np.diff(dec_dist, axis=1) - react_dist
        return self.safe_result

    def data_to_df(self):
        assert self.frame.warm_up_step >= 0, "warm_up_step必须大于等于0！"
        """环形边界一个车辆轨迹拆分为多段，id加后缀_x"""
        dict_ = {"Frame_ID": [], "v_ID": [], "Local_xVelocity": [], "Preceding_ID": [], "v_Length": [],
                 "Local_X": [], "gap": [], "dhw": [], "thw": [], "Local_xAcc": []}
        data_len = int(self.container.pos_data.shape[0])
        for i in range(self.frame.car_num):
            for key in dict_.keys():
                temp: Optional[Iterable, object] = None
                if key == "Frame_ID":
                    temp = np.arange(self.frame.warm_up_step, self.frame.sim_step).tolist()
                elif key == "v_ID":
                    count = 0
                    dict_["v_ID"].extend([i] * data_len)
                    dict_["Preceding_ID"].extend([(i + 1) if (i + 1 != self.frame.car_num) else 0] * data_len)
                    for _, temp_, _, _ in self.data_shear(self.container.pos_data, index=i):
                        dict_["Local_X"].extend(temp_ + count * self.frame.lane_length)
                        count += 1
                    continue
                elif key == "Local_xVelocity":
                    temp = self.container.speed_data[:, i]
                elif key == "Local_xAcc":
                    temp = self.container.acc_data[:, i]
                elif key == "v_Length":
                    temp = [self.frame.car_length] * data_len
                elif key == "gap":
                    temp = self.container.gap_data[:, i]
                elif key == "dhw":
                    temp = self.container.dhw_data[:, i]
                elif key == "thw":
                    temp = self.container.thw_data[:, i]
                if temp is not None:
                    dict_[key].extend(temp)
        self.df = pd.DataFrame(dict_)
        return self.df

    def data_shear(self, data, index=-1):
        time_ = np.arange(self.frame.warm_up_step, self.frame.sim_step) * self.frame.dt

        for j in range(self.frame.car_num):
            if index >= 0 and index != j:
                continue
            temp_ = data[:, j]
            return_index = list(np.where(np.diff(temp_) < 0)[0])
            return_index.insert(0, 0)
            if len(return_index) == 1:
                yield time_, temp_, j, [0, len(temp_)]
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
                    yield time__, temp__, j, pos


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
