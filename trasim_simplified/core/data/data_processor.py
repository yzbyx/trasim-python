# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:18
# @Author : yzbyx
# @File : data_processor.py
# @Software : PyCharm
from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd

try:
    from tools.info import Info as C_Info
except ImportError:
    from trasim_simplified.core.data.data_container import Info as C_Info


class DataProcessor:
    @staticmethod
    def print(result: Union[dict, tuple]):
        if isinstance(result, tuple):
            for res in result:
                for r in res.items():
                    print(r)
        else:
            for result in result.items():
                print(result)
        print()

    @staticmethod
    def circle_kqv_cal(
        df: pd.DataFrame, lane_length, lane_id=None, pos_range=None, time_range=None
    ):
        assert df[C_Info.v] is not None, "调用本函数须使用record函数记录速度数据"
        if lane_id is not None:
            df = df[df[C_Info.lane_add_num] == lane_id]
        if pos_range is not None:
            df = df[(df[C_Info.x] >= pos_range[0]) & (df[C_Info.x] < pos_range[1])]
        if time_range is not None:
            df = df[
                ((df[C_Info.time] >= time_range[0]) & (df[C_Info.time] < time_range[1]))
            ]

        avg_speed = np.mean(df.groupby(by=[C_Info.step]).mean()[C_Info.v])

        # avg_speed = df[C_Info.v].mean()
        avg_k_by_car_num_lane_length = len(df[C_Info.id].unique()) / lane_length * 1000
        avg_q_by_v_k = avg_speed * 3.6 * avg_k_by_car_num_lane_length

        return {
            Info.avg_k_by_car_num_lane_length: avg_k_by_car_num_lane_length,
            Info.avg_q_by_v_k: avg_q_by_v_k,
            Info.avg_speed: avg_speed,
        }

    @staticmethod
    def aggregate(df: pd.DataFrame, lane_id: int, lane_length: float):
        """集计指标计算"""
        if lane_id >= 0:
            df = df[df[C_Info.lane_add_num] == lane_id]
        aggregate_all_result = {}
        # 由于总车辆数恒定，因此直接对所有车辆数据求平均是可行的
        group = df.groupby(by=[C_Info.id]).mean()
        avg_acc, avg_speed, avg_gap, avg_dv, avg_dhw, avg_thw = (
            np.mean(group[C_Info.a]),
            np.mean(group[C_Info.v]),
            np.mean(group[C_Info.gap]),
            np.mean(group[C_Info.dv]),
            np.mean(group[C_Info.dhw]),
            np.mean(group[C_Info.thw]),
        )
        # TODO：标准差这样做有待商榷
        group = df.groupby(by=[C_Info.id]).std()
        std_acc, std_speed, std_gap, std_dv, std_dhw, std_thw = (
            np.std(group[C_Info.a]),
            np.std(group[C_Info.v]),
            np.std(group[C_Info.gap]),
            np.std(group[C_Info.dv]),
            np.std(group[C_Info.dhw]),
            np.std(group[C_Info.thw]),
        )
        avg_q_by_thw = 3600 / avg_thw
        avg_k_by_dhw = 1000 / avg_dhw
        avg_v_q_div_k_by_thw_dhw = avg_q_by_thw / avg_k_by_dhw / 3.6

        harmonic_avg_speed = 1 / np.mean(1 / df[C_Info.v])

        avg_k_by_car_num_lane_length = (
            len(df)
            / (lane_length * (df[C_Info.step].max() - df[C_Info.step].min()))
            * 1000
        )
        avg_q_by_v_k = avg_speed * 3.6 * avg_k_by_car_num_lane_length

        aggregate_all_result.update(
            {
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
                Info.avg_k_by_car_num_lane_length: avg_k_by_car_num_lane_length,
            }
        )
        return aggregate_all_result

    @staticmethod
    def aggregate_as_detect_loop(
        df: pd.DataFrame,
        lane_id: int,
        lane_length: float,
        pos: float,
        width: float,
        dt: float,
        d_step: int,
        step_range: Sequence[int, int] = None,
    ):
        """
        以传感线圈的方式检测交通参数（HCM的平均速度定义）

        :param df: 车辆的基础数据
        :param lane_id: 车道ID
        :param lane_length: 道路长度 [m], 开边界一般就填大数
        :param step_range: 总的检测始末仿真时刻(两个数)
        :param pos: 传感线圈起始位置 [m]
        :param width: 传感线圈宽度 [m]
        :param dt: 仿真步长 [s]
        :param d_step: 每个检测周期的总仿真步
        :return: 包含顺序集计交通参数列表的字典
        """
        if lane_id >= 0:
            df = df[df[C_Info.lane_add_num] == lane_id]
        min_width = dt * np.max(df[C_Info.v])
        assert width > min_width, f"至少将传感线圈的宽度设置在{min_width}以上！"

        is_return = False  # 传感器是否返回
        end_pos = pos + width
        if end_pos > lane_length:
            is_return = True
            end_pos -= lane_length

        if not is_return:
            pos_in = df[(pos <= df[C_Info.x]) & (df[C_Info.x] <= end_pos)]
        else:
            pos_in = df[(pos <= df[C_Info.x]) | (df[C_Info.x] <= end_pos)]

        if step_range is None:
            step_range = (df[C_Info.step].min(), df[C_Info.step].max())

        time_sections = np.arange(step_range[0], step_range[1] + 1, d_step)
        time_sections_zip = zip(time_sections[:-1], time_sections[1:])

        # 计算时间平均车速
        # 每辆车的地点车速为线圈范围内速度点的平均（前提是数据点之间的采样时间间隔相同）
        time_avg_speed_list = []
        space_avg_speed_by_time_avg_speed_list = []
        q_list = []  # veh/s
        time_occ_list = []
        # HCM计算方式
        d_A_list = []
        t_A_list = []
        area_A = width * d_step * dt

        aggregate_loop_result = {}
        aggregate_Edie_result = {}

        for time_start, time_end in time_sections_zip:
            d_a = 0
            t_a = 0
            car_num_in = 0
            avg_speed_list_per_traj = []  # 每段轨迹的平均速度

            for id_ in df[C_Info.id].unique():
                single_data = pos_in[pos_in[C_Info.id] == id_]
                single_data = single_data[
                    (single_data[C_Info.time] >= time_start)
                    & (single_data[C_Info.time] < time_end)
                ]
                single_data = single_data.sort_values(by=[C_Info.step]).reset_index(
                    drop=True
                )

                for time, speed, pos in DataProcessor.data_shear(
                    single_data[C_Info.v].to_numpy(),
                    single_data[C_Info.x].to_numpy(),
                    single_data[C_Info.time].to_numpy(),
                    single_data[C_Info.step].to_numpy(),
                    shear_pos=False,
                    shear_step=True,
                ):
                    if len(time) == 0:
                        continue

                    avg_speed_list_per_traj.append(np.mean(speed))

                    pos_data = single_data[C_Info.x][pos[0] : pos[1]].to_numpy()
                    if pos_data[0] > pos_data[-1]:
                        #  如果轨迹点折返
                        d_a += lane_length - pos_data[0] + pos_data[-1]
                    else:
                        d_a += pos_data[-1] - pos_data[0]

                    t_a += time[-1] - time[0]

                    # 行驶距离补偿
                    d_a += (speed[0] + speed[-1]) / 2 * dt
                    # 行驶时间补偿
                    t_a += dt

                    car_num_in += 1
            steps_has_car = len(
                pos_in[
                    (pos_in[C_Info.time] >= time_start)
                    & (pos_in[C_Info.time] < time_end)
                ]
            )

            time_avg_speed_list.append(np.mean(avg_speed_list_per_traj))  # 时间平均车速
            # 地点车速的调和平均为空间平均车速
            space_avg_speed_by_time_avg_speed_list.append(
                1 / np.mean(1 / np.array(avg_speed_list_per_traj))
            )
            q_list.append(car_num_in / (d_step * dt))
            time_occ_list.append(steps_has_car / d_step)

            d_A_list.append(d_a)
            t_A_list.append(t_a)

        aggregate_loop_result["loop_q(veh/h)"] = (np.array(q_list) * 3600).tolist()
        aggregate_loop_result["loop_time_occ"] = time_occ_list
        aggregate_loop_result["loop_k_by_time_occ(veh/km)"] = (
            np.array(time_occ_list) / width * 1000
        ).tolist()
        aggregate_loop_result["loop_vt(m/s)"] = time_avg_speed_list
        aggregate_loop_result["loop_vt(km/h)"] = (
            np.array(time_avg_speed_list) * 3.6
        ).tolist()
        aggregate_loop_result[
            "loop_vs_by_vt(m/s)"
        ] = space_avg_speed_by_time_avg_speed_list
        aggregate_loop_result["loop_vs_by_vt(km/h)"] = (
            np.array(space_avg_speed_by_time_avg_speed_list) * 3.6
        ).tolist()

        aggregate_Edie_result["HCM_dA(m)"] = d_A_list
        aggregate_Edie_result["HCM_tA(s)"] = t_A_list
        aggregate_Edie_result["HCM_area_A(m*s)"] = area_A
        aggregate_Edie_result["HCM_qA(veh/h)"] = (
            np.array(d_A_list) / area_A * 3600
        ).tolist()
        aggregate_Edie_result["HCM_kA(veh/km)"] = (
            np.array(t_A_list) / area_A * 1000
        ).tolist()
        aggregate_Edie_result["HCM_vA(m/s)"] = (
            np.array(d_A_list) / np.array(t_A_list)
        ).tolist()
        aggregate_Edie_result["HCM_vA(km/h)"] = (
            np.array(d_A_list) / np.array(t_A_list) * 3.6
        ).tolist()

        return aggregate_loop_result, aggregate_Edie_result

    @staticmethod
    def data_shear(
        temp_=None, pos_=None, time_=None, step_=None, shear_pos=True, shear_step=True
    ):
        """可以将轨迹数据按照环形边界返回点以及缺失点分割成多段"""
        return_index = (
            list(np.where(np.diff(np.array(pos_)) < 0)[0]) if shear_pos else []
        )
        return_index_2 = (
            list(np.where(np.diff(np.array(step_)) != 1)[0]) if shear_step else []
        )
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
                temp__ = temp_[pos[0] : pos[1]]
                time__ = time_[pos[0] : pos[1]]
                yield time__, temp__, pos


class DetectLoopInfo:
    Q = "loop_q(veh/h)"
    """进入线圈区域的流量"""
    Time_Occ = "loop_time_occ"
    '"线圈区域有车的时间占比"'
    K_By_Time_Occ = "loop_k_by_time_occ(veh/km)"
    """通过时间占有率计算的密度"""
    Vt = "loop_vt(m/s)"
    """时间平均速度"""
    Vt_KPH = "loop_vt(km/h)"
    """时间平均速度 [km/h]"""
    Vs_By_Vt = "loop_vs_by_vt(m/s)"
    """空间平均速度，时间平均速度的调和平均"""
    Vs_By_Vt_KPH = "loop_vs_by_vt(km/h)"
    """空间平均速度 [km/h]"""
    HCM_dA = "HCM_dA(m)"
    """时空范围内车辆行驶总距离"""
    HCM_tA = "HCM_tA(s)"
    """时空范围内车辆行驶总时长"""
    HCM_A = "HCM_area_A(m*s)"
    """时空范围的面积"""
    HCM_qA = "HCM_qA(veh/h)"
    """HCM流量"""
    HCM_kA = "HCM_kA(veh/km)"
    """HCM密度"""
    HCM_vA = "HCM_vA(m/s)"
    """HCM速度"""
    HCM_vA_KPH = "HCM_vA(km/h)"
    """HCM速度 [km/h]"""


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


if __name__ == "__main__":
    print(Info.get_all_info())
