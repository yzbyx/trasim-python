# -*- coding = uft-8 -*-
# @Time : 2022-08-01 8:08
# @Author : yzbyx
# @File : dataAnalysis.py
# @Software : PyCharm
import random

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

pd.set_option('display.width', 5000)


def readData(path: str, usecols: list[str], rename_dict: dict[str: str], is_imperial=False):
    """一、数据标准化
        1. 车辆Vehicle_ID
        2. 帧Frame_ID
        3. 车道Lane_ID
        4. 路线偏移Local_Y
        5. 车辆速度v_Vel
        6. 车辆加速度v_Acc
        7. 车身长度v_Length
        8. 车辆类型v_Class
        9. 跟随车辆Follower_ID
        10. 前导车辆Leader_ID
    """
    data = pd.read_csv(path, usecols=usecols)
    data.rename(columns=rename_dict, inplace=True)
    if is_imperial:
        data[['Local_Y', 'v_Vel', 'v_Acc', 'v_Length']] = data[['Local_Y', 'v_Vel', 'v_Acc', 'v_Length']].apply(
            lambda x: x * 0.3048).round(decimals=5)
    assert len(data[data['Local_Y'].isna()]) == 0, "数据包含NaN值！"
    data.to_pickle('data/1_data_std.pkl')


def internalConsistencyBasic(need_calculate=True):
    print("\n# -----内部一致性——基础指标----- #")
    data = pd.read_pickle('data/1_data_std.pkl')

    # 车辆总数量 & 平均观测时长 & 平均行驶距离
    car_list = list(data['Vehicle_ID'].unique())
    temp1 = f"车辆总数：{len(car_list)}辆"
    print(temp1)
    temp2 = "平均观测时长：{:.2f}s".format(len(data) * 0.1 / len(car_list))
    print(temp2)

    distance = 0
    for car in car_list:
        target = data[data['Vehicle_ID'] == car]
        distance += target['Local_Y'].max() - target['Local_Y'].min()
    temp3 = "平均行驶距离：{:.2f}m".format(distance / len(car_list))
    print(temp3)

    # 缺帧分析
    car_list = data['Vehicle_ID'].unique().tolist()
    for car in car_list:
        target = data[data['Vehicle_ID'] == car]
        assert target['Frame_ID'].max() - target['Frame_ID'].min() == len(target) - 1, f'{car}缺帧'

    # 速度 & 加速度 & Jerk值 & Jerk变化次数异常分析
    # 速度异常分析
    arr_std = data[data['v_Vel'] > 55 * 1609.344 / 3600]['v_Vel'].tolist()
    arr_120 = data[data['v_Vel'] > 55 * 1609.344 * 1.2 / 3600]['v_Vel'].tolist()
    arr_neg = data[data['v_Vel'] < 0]['v_Vel'].tolist()
    print(f'限速值(m/s)：{55 * 1609.344 / 3600}')
    print(f"超出限速帧数量：{len(arr_std)}")
    print("超出限速帧比例：{:.2f}".format(len(arr_std) / len(data)))

    print(f'120%限速值(m/s)：{55 * 1609.344 * 1.2 / 3600}')
    print(f"超出120%限速帧数量：{len(arr_120)}")
    print("超出120%限速帧比例：{:.2f}".format(len(arr_120) / len(data)))

    temp = arr_std
    temp.extend(arr_neg)
    print(f'最小异常速度(m/s)：{np.min(temp)}')
    print(f'最大异常速度(m/s)：{np.max(temp)}')

    print(f'速度异常数量：{len(arr_std) + len(arr_neg)}')
    print('速度异常比例：{:.2f}%'.format((len(arr_std) + len(arr_neg)) / len(data) * 100))

    # 加速度异常分析
    arr = data[(data['v_Acc'] < -8) | data['v_Acc'] > 5]['v_Acc']
    print(f'最小异常加速度(m/s^2)：{np.min(arr)}')
    print(f'最大异常加速度(m/s^2)：{np.max(arr)}')
    print(f'加速度异常数量：{len(arr)}')
    print('加速度异常比例：{:.2f}%'.format(len(arr) / len(data) * 100))

    # Jerk值异常 & Jerk变化异常
    if need_calculate:
        data['v_Jerk'] = np.NAN
        VehicleList = data['Vehicle_ID'].unique().tolist()
        total_num = len(VehicleList)

        for i, vehicleID in enumerate(VehicleList):
            data_by_vehicle = data[data['Vehicle_ID'] == vehicleID].sort_values(by=['Frame_ID'])
            assert data_by_vehicle['Frame_ID'].max() - data_by_vehicle['Frame_ID'].min() == len(
                data_by_vehicle) - 1, '缺帧！'
            data_by_vehicle['v_Jerk'] = (data_by_vehicle['v_Acc'].shift(-1) - data_by_vehicle['v_Acc']).shift(1) * 10
            data.loc[data_by_vehicle.index, 'v_Jerk'] = data_by_vehicle['v_Jerk']
            print(f'\r正在进行第{i + 1}辆车，共{total_num}辆车...', end='')

        data['Jerk_Change'] = 0
        data['Jerk_Change_Sum'] = 0
        car_list = data['Vehicle_ID'].unique().tolist()
        total_num = len(car_list)

        for i, car in enumerate(car_list):
            data_by_vehicle = data[data['Vehicle_ID'] == car].sort_values(by=['Frame_ID'])
            data_by_vehicle['v_Jerk_Simplified'] = data_by_vehicle['v_Jerk'].apply(lambda x: -1 if x < 0 else 1)
            data_by_vehicle['Jerk_Change_temp'] = (
                    data_by_vehicle['v_Jerk_Simplified'].shift(-1) + data_by_vehicle['v_Jerk_Simplified']).shift(1)
            data_by_vehicle.loc[data_by_vehicle[data_by_vehicle['Jerk_Change_temp'] == 0].index, 'Jerk_Change'] = 1

            temp = data_by_vehicle.copy()
            data_by_vehicle['Jerk_Change_Sum'] = 0
            for j in range(5):
                data_by_vehicle['Jerk_Change_Sum'] += temp['Jerk_Change'].shift(j + 1).fillna(0)
            for j in range(5):
                data_by_vehicle['Jerk_Change_Sum'] += temp['Jerk_Change'].shift(-j).fillna(0)

            data.loc[data_by_vehicle.index, ['Jerk_Change', 'Jerk_Change_Sum']] = data_by_vehicle[
                ['Jerk_Change', 'Jerk_Change_Sum']]

            print(f'\r正在进行第{i + 1}辆车，共{total_num}辆车...', end='')
        print("\r", end="")
        data.to_pickle('data/6_jerk_and_change_result.pkl')

    data = pd.read_pickle('data/6_jerk_and_change_result.pkl')
    arr = data[data['v_Jerk'].abs() > 15]['v_Jerk']
    if len(arr) != 0:
        print('最小异常加速度变化率(m/s^3)：{:.2f}'.format(arr.min()))
        print('最大异常加速度变化率(m/s^3)：{:.2f}'.format(arr.max()))
    print('加速度变化率异常数量：{}'.format(len(arr)))
    print('加速度变化率异常比例：{:.2f}'.format(len(arr) / len(data)))
    arr = data[data['Jerk_Change_Sum'] > 1]['Jerk_Change_Sum']
    if len(arr) != 0:
        print(f'最大异常加速度变化率1s变号次数：{arr.max()}')
    print(f'加速度变化率1s变化异常数量：{len(arr)}')
    print('加速度变化率1s变化异常比例：{:.2f}'.format(len(arr) / len(data)))


def platoonConsistencyBasic(need_calculate=True):
    print("\n# -----队列一致性——基础指标----- #")
    data: pd.DataFrame = pd.read_pickle('data/1_data_std.pkl')

    # 预处理——统计车辆对
    if need_calculate:
        data['Leader'] = data['Leader'].astype(dtype=np.int64)
        total_lane_list = data['Lane_ID'].unique()
        total_num = 1  # 所有跟驰对的ID

        Current_index = []
        Leader_index = []
        Pair_ID = []
        Gap = []
        Space_Headway = []
        Time_Headway = []
        TTC = []
        Delta_V = []

        Current_Y = []
        Current_V = []
        Current_Acc = []
        Leader_Y = []
        Leader_V = []
        Leader_Acc = []

        for ii, laneID in enumerate(total_lane_list):
            tempIndex = []
            df = data[data['Lane_ID'] == laneID]
            car_id_list = df['Vehicle_ID'].unique()
            lossFrame = 0
            num = len(car_id_list)
            for j, v in enumerate(car_id_list):
                print(f'\rLane:{ii + 1}/{len(total_lane_list)}\t第{j}/{num}辆车...', end='')
                car_data = df[df['Vehicle_ID'] == v].sort_values(by=['Frame_ID'], ascending=True)
                # 存放单个车辆的所有跟驰片段
                leaderPre = 0
                preFrame = -10
                for i, index in enumerate(car_data.index):
                    dataByFrame = car_data.take([i])
                    frameID = dataByFrame['Frame_ID'].values[0]
                    leader = dataByFrame['Leader'].values[0]
                    leaderByFrame = df[(df['Frame_ID'] == frameID) & (df['Vehicle_ID'] == leader)]
                    if len(leaderByFrame) == 0:  # 前车数据缺失
                        if leader != 0:
                            lossFrame += 1
                            # print(f'\r{v}对应前车{leader}帧数据{frameID}缺失！共缺{lossFrame}', end='')
                        leaderPre = leader
                        preFrame = frameID
                        if len(tempIndex) != 0:
                            tempIndex.clear()
                            total_num += 1
                        continue
                    if preFrame != frameID - 1 or leaderPre != leader:  # 前方帧断连或前方换车
                        tempIndex.clear()
                        total_num += 1

                    leaderIndex = leaderByFrame.index.values[0]
                    leaderY = leaderByFrame['Local_Y'].values[0]
                    currentY = dataByFrame['Local_Y'].values[0]
                    leaderL = leaderByFrame['v_Length'].values[0]
                    gap = leaderY - currentY - leaderL
                    space_headway = leaderY - currentY
                    leaderV = leaderByFrame['v_Vel'].values[0]
                    currentV = dataByFrame['v_Vel'].values[0]
                    deltaV = currentV - leaderV
                    time_headway = space_headway / currentV if currentV != 0 else np.inf
                    ttc = gap / deltaV if deltaV != 0 else np.inf

                    tempIndex.append(index)

                    Current_index.append(index)
                    Pair_ID.append(total_num)
                    Leader_index.append(leaderIndex)
                    Gap.append(gap)
                    Space_Headway.append(space_headway)
                    Time_Headway.append(time_headway)
                    TTC.append(ttc)
                    Delta_V.append(deltaV)

                    Current_Y.append(currentY)
                    Current_V.append(currentV)
                    Current_Acc.append(dataByFrame['v_Acc'].values[0])
                    Leader_Y.append(leaderY)
                    Leader_V.append(leaderV)
                    Leader_Acc.append(leaderByFrame['v_Acc'].values[0])

                    leaderPre = leader
                    preFrame = frameID

        df_new = pd.DataFrame(index=Current_index, data={'Pair_ID': Pair_ID,
                                                         'Leader_Index': Leader_index,
                                                         'Current_Y': Current_Y,
                                                         'Current_V': Current_V,
                                                         'Current_Acc': Current_Acc,
                                                         'Leader_Y': Leader_Y,
                                                         'Leader_V': Leader_V,
                                                         'Leader_Acc': Leader_Acc,
                                                         'Gap': Gap,
                                                         'Space_Headway': Space_Headway,
                                                         'Time_Headway': Time_Headway,
                                                         'TTC': TTC,
                                                         'Delta_V': Delta_V})
        df_new.to_pickle('data/3_follow_pair.pkl')
        print("\r", end="")
    # 车辆对数量 & 平均跟驰时长 & 平均跟驰距离
    df_new = pd.read_pickle('data/3_follow_pair.pkl')
    pair_list = list(df_new['Pair_ID'].unique())
    print(f"车辆对数量：{len(pair_list)}对")
    print("平均跟驰时长：{:.2f}s".format(np.average(df_new.groupby(by=['Pair_ID']).count().reset_index()['Gap']) / 10))

    distance = 0
    for pair_id in pair_list:
        target = data.loc[df_new[df_new['Pair_ID'] == pair_id].index]
        distance += target['Local_Y'].max() - target['Local_Y'].min()
    print("平均跟驰距离：{:.2f}m".format(distance / len(pair_list)))

    frame = set(df_new.index.to_list()) | set(list(df_new['Leader_Index']))
    print("跟驰数据帧占比：{:.2f}%".format(len(frame) / len(data) * 100))

    # 净间距平均 & 车头间距平均 & 车头时距平均
    print("净间距平均：{:.2f}m".format(np.average(df_new['Gap'])))
    print("车头间距平均：{:.2f}m".format(np.average(df_new['Space_Headway'])))
    print("车头时距平均：{:.2f}s".format(np.average(df_new[~np.isinf(df_new['Time_Headway'])]['Time_Headway'])))

    # 间距异常分析
    arr1 = df_new[df_new['Gap'] < 0]['Gap']
    arr2 = df_new[df_new['Space_Headway'] < 0]['Space_Headway']
    print(f'最小异常净间距(m)：{np.min(arr1)}')
    print(f'净间距异常数量：{len(arr1)}')
    print('净间距异常比例：{:.2f}%'.format(len(arr1) / len(df_new) * 100))

    print(f'最小异常车头间距(m)：{np.min(arr2)}')
    print(f'车头间距异常数量：{len(arr2)}')
    print('车头间距异常比例：{:.2f}%'.format(len(arr2) / len(df_new) * 100))


def internalConsistency(need_calculate=True):
    print("\n# -----内部一致性——一致性指标----- #")
    data: pd.DataFrame = pd.read_pickle('data/1_data_std.pkl')

    # 预处理——积分计算
    if need_calculate:
        car_list = list(data['Vehicle_ID'].unique())
        data['Y_Integral'] = 0
        data['Y_Difference'] = np.NAN
        data['V_Integral'] = 0
        data['V_Difference'] = np.NAN
        for j, car in enumerate(car_list):
            target = data[data['Vehicle_ID'] == car].sort_values(by=['Frame_ID']).copy()
            assert target['Frame_ID'].max() - target['Frame_ID'].min() == len(target) - 1, f'{car}缺帧！'

            target.loc[target.index[0], 'v_Acc'] = 0
            target_added: pd.DataFrame = target.copy()
            for i, index in enumerate(target.index):  # 矩形积分
                target_added['V_Integral'] += target['v_Acc'].shift(i).fillna(0) * 0.1
            init_vel = target_added.loc[target_added.index[0], 'v_Vel']
            target_added['V_Integral'] += init_vel
            target_added['V_Difference'] = target_added['v_Vel'] - target_added['V_Integral']

            target.loc[target.index[0], 'v_Vel'] = 0
            for i, index in enumerate(target.index):  # 矩形积分
                target_added['Y_Integral'] += target['v_Vel'].shift(i).fillna(0) * 0.1
            init_Local_Y = target_added.loc[target_added.index[0], 'Local_Y']
            target_added['Y_Integral'] += init_Local_Y
            target_added['Y_Difference'] = target_added['Local_Y'] - target_added['Y_Integral']

            data.loc[target_added.index, ['Y_Integral', 'Y_Difference', 'V_Integral', 'V_Difference']] = target_added[
                ['Y_Integral', 'Y_Difference', 'V_Integral', 'V_Difference']]
            print(f'\r已完成{j + 1}辆，共{len(car_list)}辆', end='')
        data.to_pickle('data/4_integral_data.pkl')
        print("\r", end="")

    # 行驶距离一致性 & 速度变化一致性
    data = pd.read_pickle('data/4_integral_data.pkl')
    mean_bias = np.average(data['Y_Difference'])
    print('行驶距离一致性——偏差均值：{:.2f}m'.format(mean_bias))

    RMSEs = []
    RMSPEs = []
    car_list = list(data['Vehicle_ID'].unique())
    for car in car_list:
        target = data[data['Vehicle_ID'] == car].sort_values(by=['Frame_ID'])
        # print(target['Local_Y'].isna().values.any())
        # assert target['Local_Y'].isna().values.any() is False, "Local_Y存在NaN值"
        RMSPE = np.sqrt(np.average((target['Y_Difference'] / target['Local_Y']) ** 2))
        RMSE = np.sqrt(np.average(target['Y_Difference'] ** 2))
        RMSPEs.append(RMSPE)
        RMSEs.append(RMSE)
    print('行驶距离一致性——均方根误差均值（RMSE）：{:.2f}'.format(np.average(RMSEs)))
    print('行驶距离一致性——均方根百分比误差均值（RMSPE）：{:.2f}%'.format(np.average(RMSPEs) * 100))

    data = pd.read_pickle('data/4_integral_data.pkl')
    mean_bias = np.average(data['V_Difference'])
    print('速度变化一致性——偏差均值：{:.2f}m/s'.format(mean_bias))

    RMSEs = []
    RMSPEs = []
    car_list = list(data['Vehicle_ID'].unique())
    for car in car_list:
        target = data[data['Vehicle_ID'] == car].sort_values(by=['Frame_ID'])
        # TODO: RMSPE的分母可能为0
        target['v_Vel'] += 1e-10
        RMSPE = np.sqrt(np.average((target['V_Difference'] / target['v_Vel']) ** 2))
        RMSE = np.sqrt(np.average(target['V_Difference'] ** 2))
        RMSPEs.append(RMSPE)
        RMSEs.append(RMSE)
    print('速度变化一致性——均方根误差均值（RMSE）：{:.2f}'.format(np.average(RMSEs)))
    print('速度变化一致性——均方根百分比误差均值（RMSPE）：{:.2f}%'.format(np.average(RMSPEs) * 100))


def platoonConsistency():
    print("\n# -----队列一致性——一致性指标----- #")
    data: pd.DataFrame = pd.read_pickle('data/4_integral_data.pkl')
    data_pair: pd.DataFrame = pd.read_pickle('data/3_follow_pair.pkl')

    # 车头间距一致性
    pair_list = list(data_pair['Pair_ID'].unique())
    Ds = []
    RMSEs = []
    RMSPEs = []
    for pair in pair_list:
        target_pair = data_pair[data_pair['Pair_ID'] == pair]
        target = data.loc[target_pair.index].reset_index()
        leader = data.loc[target_pair['Leader_Index']].reset_index()

        Delta_Y = leader['Local_Y'] - target['Local_Y']
        Space_Difference = Delta_Y - (leader['Y_Integral'] - target['Y_Integral'])
        Ds.extend(list(Space_Difference))
        RMSEs.append(np.sqrt(np.average(Space_Difference ** 2)))
        RMSPEs.append(np.sqrt(np.average((Space_Difference / Delta_Y) ** 2)))
    print('车头间距——差值均值：{:.2f}m'.format(np.average(Ds)))
    print('车头间距——均方根误差均值（RMSE）：{:.2f}'.format(np.average(RMSEs)))
    print('车头间距——均方根百分比误差（RMSPE）：{:.2f}%'.format(np.average(RMSPEs) * 100))


def spectrumAnalysis():
    data: pd.DataFrame = pd.read_pickle('data/6_jerk_and_change_result.pkl')

    # 速度频谱图
    x, y, xf, yf = __spectrumAnalysis(data, 'v_Vel')

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    font_size = 16

    ax: plt.Axes = axs[0]
    ax.plot(x, y)
    ax.set_title('原始波形', fontsize=font_size)
    ax.set_xlabel('时间 (s)', fontsize=font_size)
    ax.set_ylabel('速度 (m/s)', fontsize=font_size)

    ax: plt.Axes = axs[1]
    ax.plot(xf, yf)
    ax.set_title('速度频谱图', fontsize=font_size)
    ax.set_xlabel('频率 (Hz)', fontsize=font_size)
    ax.set_ylabel('振幅', fontsize=font_size)

    fig.savefig("figure/速度频谱图.svg", bbox_inches='tight')
    plt.close()

    # 加速度频谱图
    x, y, xf, yf = __spectrumAnalysis(data, 'v_Acc')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    font_size = 16

    ax: plt.Axes = axs[0]
    ax.plot(x, y)
    ax.set_title('原始波形', fontsize=font_size)
    ax.set_xlabel('时间 (s)', fontsize=font_size)
    ax.set_ylabel('加速度 (m/s^2)', fontsize=font_size)

    ax: plt.Axes = axs[1]
    ax.plot(xf, yf)
    ax.set_title('加速度频谱图', fontsize=font_size)
    ax.set_xlabel('频率 (Hz)', fontsize=font_size)
    ax.set_ylabel('振幅', fontsize=font_size)

    fig.savefig("figure/加速度频谱图.svg", bbox_inches='tight')
    plt.close()

    # Jerk频谱图
    x, y, xf, yf = __spectrumAnalysis(data, 'v_Jerk')

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    font_size = 16

    ax: plt.Axes = axs[0]
    ax.plot(x, y)
    ax.set_title('原始波形', fontsize=font_size)
    ax.set_xlabel('时间 (s)', fontsize=font_size)
    ax.set_ylabel('Jerk (m/s^3)', fontsize=font_size)

    ax: plt.Axes = axs[1]
    ax.plot(xf, yf)
    ax.set_title('Jerk频谱图', fontsize=font_size)
    ax.set_xlabel('频率 (Hz)', fontsize=font_size)
    ax.set_ylabel('振幅', fontsize=font_size)

    fig.savefig("figure/Jerk频谱图.svg", bbox_inches='tight')
    plt.close()


def positionDifferenceAnalysis(sample_car=5):
    data: pd.DataFrame = pd.read_pickle('data/6_jerk_and_change_result.pkl')

    # 位置差分与速度对比
    sample_car = sample_car
    target = data[data['Vehicle_ID'] == sample_car].sort_values(by=['Frame_ID'])
    if len(target) == 0:
        car_list = data['Vehicle_ID'].unique().to_list()
        sample_car = random.choice(car_list)
        target = data[data['Vehicle_ID'] == sample_car].sort_values(by=['Frame_ID'])

    x = target['Frame_ID'] * 0.1
    original_v = target['v_Vel']
    diff_v = (target['Local_Y'].shift(-1) - target['Local_Y']).shift(1) * 10

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    font_size = 16

    ax = axs[0]
    ax.plot(x, original_v, 'r', label='处理后速度')
    ax.plot(x, diff_v, label='位置差分速度')
    ax.legend(fontsize=font_size - 2, frameon=False)
    ax.set_title(f'Vehicle_ID = {sample_car}', fontsize=font_size)
    ax.set_xlabel('时间 (s)', fontsize=font_size)
    ax.set_ylabel('速度 (m/s)', fontsize=font_size)

    ax = axs[1]
    ax.plot(x, diff_v - original_v, label='位置差分速度-处理后速度')
    ax.legend(fontsize=font_size - 2, frameon=False)
    ax.set_title(f'Vehicle_ID = {sample_car}', fontsize=font_size)
    ax.set_xlabel('时间 (s)', fontsize=font_size)
    ax.set_ylabel('速度差 (m/s)', fontsize=font_size)

    fig.savefig("figure/位置差分速度与处理后速度对比.svg", bbox_inches='tight')


def __spectrumAnalysis(data: pd.DataFrame, target: str):
    data[target].fillna(0)

    fs = 10

    y = data[target].fillna(np.average(data[target])).tolist()

    N = len(y)
    x = np.arange(0, N * 0.1, 1 / fs)  # 频率个数

    yf = fft(y)  # 快速傅里叶变换
    xf = fftfreq(N, 1 / fs)[: N // 2]

    return x, y, xf, 2.0 / N * np.abs(yf[:N // 2])


if __name__ == '__main__':
    readData(path="trajectories-0750am-0805am-1000frame.csv",
             usecols=['Vehicle_ID', 'Frame_ID', 'Lane_ID', 'Local_Y', 'v_Vel', 'v_Acc', 'v_Length',
                      'v_Class', 'Following', 'Preceeding'],
             rename_dict={'Preceeding': 'Leader', 'Following': 'Follower'})
    internalConsistencyBasic(need_calculate=True)
    platoonConsistencyBasic(need_calculate=True)
    internalConsistency(need_calculate=True)
    platoonConsistency()
    spectrumAnalysis()
    positionDifferenceAnalysis()
