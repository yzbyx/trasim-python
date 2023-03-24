# -*- coding = uft-8 -*-
# @Time : 2022-04-07 16:58
# @Author : yzbyx
# @File : hysteresisJudge.py
# @Software : PyCharm
import numpy as np
from util.area import getAreaByOrderedPoints

# def getHysteresisArea(v_data, other_data):
#     print('不要使用该方法！')
#     if len(v_data) != len(other_data):
#         raise ValueError('数据长度不一致！')
#
#     area = 0
#     for i in range(len(v_data) - 1):
#         area += (v_data[i] - v_data[i + 1]) * (other_data[i] + other_data[i + 1]) / 2
#
#     return area


def hysteresisJudge_revise(decData, accData, dec2acc: bool, errorJudge=True, leaderL=5):
    if (len(decData[1]) < 10 or len(accData[1]) < 10) and errorJudge:  # 数据截取过短
        return -4, [], []

    minV = max(min(decData[1]), min(accData[1]))
    maxV = min(max(decData[1]), max(accData[1]))
    if maxV <= minV and errorJudge:  # 加速和减速数据未重叠，主要由于数据过短
        return -3, [], []
    if maxV - minV < 10 and errorJudge:  # 速度差不足
        return -6, [], []

    decIndex = np.where((decData[1] >= minV) & (decData[1] <= maxV))[0]
    accIndex = np.where((accData[1] >= minV) & (accData[1] <= maxV))[0]
    if (len(decIndex) < 10 or len(accIndex) < 10) and errorJudge:  # 有效数据过短
        return -5, [], []

    vDec = []
    gapDec = []
    qDec = []
    vAcc = []
    gapAcc = []
    qAcc = []

    if len(np.where(decData[0] > 120)[0]) != 0 or len(np.where(accData[0] > 120)[0]) != 0:
        raise ValueError('存在大于120m')

    for i in decIndex:
        vDec.append(decData[1][i])
        gapDec.append(decData[0][i])
        qDec.append(3600 * vDec[-1] / (gapDec[-1] + 5))
    for i in accIndex:
        vAcc.append(accData[1][i])
        gapAcc.append(accData[0][i])
        qAcc.append(3600 * vAcc[-1] / (gapAcc[-1] + 5))
    if dec2acc:
        vDec.extend(vAcc)
        gapDec.extend(gapAcc)
        qDec.extend(qAcc)
        gAvg = - getAreaByOrderedPoints(vDec, gapDec) / (maxV - minV)
        qAvg = - getAreaByOrderedPoints(vDec, qDec) / (maxV - minV)
        vAvg = np.average(vDec)
        gapAvg = np.average(gapDec)
        vMaxDelta = maxV - minV
        gapMaxDelta = max(gapDec) - min(gapDec)
    else:
        vAcc.extend(vDec)
        gapAcc.extend(gapDec)
        qAcc.extend(qDec)
        gAvg = - getAreaByOrderedPoints(vAcc, gapAcc) / (maxV - minV)
        qAvg = - getAreaByOrderedPoints(vAcc, qAcc) / (maxV - minV)
        vAvg = np.average(vAcc)
        gapAvg = np.average(gapAcc)
        vMaxDelta = maxV - minV
        gapMaxDelta = max(gapAcc) - min(gapAcc)

    # 减速面积计算
    areaDec = 0
    areaDecQ = 0
    preIndex = decIndex[0]
    for index in decIndex[1:]:
        if index - 1 == preIndex:  # 判断数据点是否连续，其实不连续是有问题的
            areaDec += ((decData[0][preIndex] + decData[0][index]) / 2) * \
                       (decData[1][index] - decData[1][preIndex])
        preIndex = index
    # 加速面积计算
    areaAcc = 0
    preIndex = accIndex[0]
    for index in accIndex[1:]:
        if index - 1 == preIndex:  # 判断数据点是否连续
            areaAcc += ((accData[0][preIndex] + accData[0][index]) / 2) * \
                       (accData[1][index] - accData[1][preIndex])
        preIndex = index

    if (areaAcc < 0 or areaDec > 0) and errorJudge:  # 数据过于奇怪，主要由于能量波峰识别精度有限，且通常由于数据较短导致的问题
        return -2, [], []

    if qAvg > 0:
        return -1, [accIndex, decIndex], [gAvg, qAvg, minV, maxV, vAvg, gapAvg, vMaxDelta, gapMaxDelta]  # 反迟滞
    elif qAvg < -300:
        return 2, [accIndex, decIndex], [gAvg, qAvg, minV, maxV, vAvg, gapAvg, vMaxDelta, gapMaxDelta]  # strong level
    elif qAvg < -50:
        return 1, [accIndex, decIndex], [gAvg, qAvg, minV, maxV, vAvg, gapAvg, vMaxDelta, gapMaxDelta]  # weak level
    elif qAvg <= 0:
        return 0, [accIndex, decIndex], [gAvg, qAvg, minV, maxV, vAvg, gapAvg, vMaxDelta, gapMaxDelta]  # negligible level


def hysteresisJudge_revise_calibrate(data, dec2acc: bool, errorJudge=True, leaderL=5):
    """data: [gap, speed]"""
    vDec = []
    gapDec = []
    qDec = []
    vAcc = []
    gapAcc = []
    qAcc = []

    gap: list = list(data[0])
    speed: list = list(data[1])

    if dec2acc:
        minV = min(speed)
        index = speed.index(minV)
        part1 = speed[:index]
        part2 = speed[index:]
        maxPart1 = max(part1)
        maxPart2 = max(part2)
        maxV = min(maxPart1, maxPart2)
        if maxPart1 <= maxPart2:
            offset = 0
            part2.reverse()
            for value in part2:
                if value < maxV:
                    break
                offset += 1
            if offset == 0:
                offset = 1
            speed_revised = speed[:-offset]
            gap_revised = gap[:-offset]
            decIndex = list(range(0, index))
            accIndex = list(range(index, len(speed_revised)))
        else:
            offset = 0
            for value in part1:
                if value < maxV:
                    break
                offset += 1
            if offset == 0:
                offset = 1
            speed_revised = speed[offset:]
            gap_revised = gap[offset:]
            decIndex = list(range(offset, index))
            accIndex = list(range(index, len(speed)))
    else:
        maxV = max(speed)
        index = speed.index(maxV)
        part1 = speed[:index]
        part2 = speed[index:]
        minPart1 = min(part1)
        minPart2 = min(part2)
        minV = max(minPart1, minPart2)
        if minPart1 <= minPart2:
            offset = 0
            for value in part1:
                if value > minV:
                    break
                offset += 1
            if offset == 0:
                offset = 1
            speed_revised = speed[offset:]
            gap_revised = gap[offset:]
            accIndex = list(range(offset, index))
            decIndex = list(range(index, len(speed)))
        else:
            offset = 0
            part2.reverse()
            for value in part2:
                if value > maxV:
                    break
                offset += 1
            if offset == 0:
                offset = 1
            speed_revised = speed[:- offset]
            gap_revised = gap[:- offset]
            accIndex = list(range(0, index))
            decIndex = list(range(index, len(speed_revised)))

    q_revised = 3600 * np.array(speed_revised) / (np.array(gap_revised) + leaderL)
    try:
        gAvg = - getAreaByOrderedPoints(speed_revised, gap_revised) / (maxV - minV)
        qAvg = - getAreaByOrderedPoints(speed_revised, q_revised) / (maxV - minV)
    except RuntimeError as e:
        print(e)
        gAvg = np.nan
        qAvg = np.nan

    if qAvg > 0:
        return -1, [accIndex, decIndex], [gAvg, qAvg, minV, maxV]  # 反迟滞
    elif qAvg < -300:
        return 2, [accIndex, decIndex], [gAvg, qAvg, minV, maxV]  # strong level
    elif qAvg < -50:
        return 1, [accIndex, decIndex], [gAvg, qAvg, minV, maxV]  # weak level
    elif qAvg <= 0:
        return 0, [accIndex, decIndex], [gAvg, qAvg, minV, maxV]  # negligible level


def hysteresisJudge(decData, accData, errorJudge=True, leaderL=5):
    """DecData -> 减速部分gap-speed数据；AccData -> 加速部分gap-speed数据"""
    if (len(decData[1]) < 10 or len(accData[1]) < 10) and errorJudge:  # 数据截取过短
        return -4, [], []

    minV = max(min(decData[1]), min(accData[1]))
    maxV = min(max(decData[1]), max(accData[1]))
    if maxV <= minV:  # 加速和减速数据未重叠，主要由于数据过短
        return -3, [], []
    if maxV - minV < 10 and errorJudge:  # 速度差不足
        return -6, [], []

    decIndex = np.where((decData[1] >= minV) & (decData[1] <= maxV))[0]
    accIndex = np.where((accData[1] >= minV) & (accData[1] <= maxV))[0]
    if (len(decIndex) < 10 or len(accIndex) < 10) and errorJudge:  # 有效数据过短
        return -5, [], []

    # 减速面积计算
    areaDec = 0
    areaDecQ = 0
    preIndex = decIndex[0]
    for index in decIndex[1:]:
        if index - 1 == preIndex:  # 判断数据点是否连续，其实不连续是有问题的
            areaDec += ((decData[0][preIndex] + decData[0][index]) / 2) * \
                       (decData[1][index] - decData[1][preIndex])
        preIndex = index
    # 加速面积计算
    areaAcc = 0
    preIndex = accIndex[0]
    for index in accIndex[1:]:
        if index - 1 == preIndex:  # 判断数据点是否连续
            areaAcc += ((accData[0][preIndex] + accData[0][index]) / 2) * \
                       (accData[1][index] - accData[1][preIndex])
        preIndex = index

    if areaAcc < 0 or areaDec > 0:  # 数据过于奇怪，主要由于能量波峰识别精度有限，且通常由于数据较短导致的问题
        return -2, [], []

    # areaDelta = areaAcc + areaDec
    # gapAvg = areaDelta / (maxV - minV)

    # 流量最大差值计算
    QDelta = []
    GapDelta = []
    for index in decIndex:  # 减速区域循环
        v = decData[1][index]
        gap = decData[0][index]
        QDec = 3600 * v / (gap + leaderL)
        accV_0 = -1
        accGap_0 = -1
        accV_1 = -1
        accGap_1 = -1
        loc = np.where(accData[1] <= v)[0]  # 左侧加速最近点寻找
        if len(loc) > 0:
            accV_0 = accData[1][loc[-1]]
            accGap_0 = accData[0][loc[-1]]
        loc = np.where(accData[1] > v)[0]  # 右侧加速最近点寻找
        if len(loc) > 0:
            accV_1 = accData[1][loc[0]]
            accGap_1 = accData[0][loc[0]]
        if accV_0 != -1 and accV_1 != -1:  # 两边均有数据
            accV = (accV_0 + accV_1) / 2
            accGap = (accGap_0 + accGap_1) / 2
        else:
            accV = max(accV_0, accV_1)
            accGap = max(accGap_0, accGap_1)
        QDelta.append(QDec - 3600 * accV / (accGap + leaderL))
        GapDelta.append(accGap - gap)
    for index in accIndex:  # 减速区域循环
        v = accData[1][index]
        gap = accData[0][index]
        QAcc = 3600 * v / (gap + leaderL)
        decV_0 = -1
        decGap_0 = -1
        decV_1 = -1
        decGap_1 = -1
        loc = np.where(decData[1] <= v)[0]
        if len(loc) > 0:
            decV_0 = decData[1][loc[-1]]
            decGap_0 = decData[0][loc[-1]]
        loc = np.where(decData[1] > v)[0]
        if len(loc) > 0:
            decV_1 = decData[1][loc[0]]
            decGap_1 = decData[0][loc[0]]
        if decV_0 != -1 and decV_1 != -1:  # 两边均有数据
            # FIXME
            decV = (decV_0 + decV_1) / 2
            decGap = (decGap_0 + decGap_1) / 2
        else:
            decV = max(decV_0, decV_1)
            decGap = max(decGap_0, decGap_1)
        QDelta.append(3600 * decV / (decGap + leaderL) - QAcc)
        GapDelta.append(gap - decGap)
    QDelta = sum(QDelta) / len(QDelta)
    gapAvg = sum(GapDelta) / len(GapDelta)

    if QDelta < 0:
        return -1, [accIndex, decIndex], [gapAvg, QDelta, minV, maxV]  # 反迟滞
    elif QDelta > 300:
        return 2, [accIndex, decIndex], [gapAvg, QDelta, minV, maxV]  # strong level
    elif QDelta > 50:
        return 1, [accIndex, decIndex], [gapAvg, QDelta, minV, maxV]  # weak level
    elif QDelta >= 0:
        return 0, [accIndex, decIndex], [gapAvg, QDelta, minV, maxV]  # negligible level


# if __name__ == '__main__':
#     # hysteresisJudge_revise([[0, 1, 1, 0], [0, 0, 1, 1]], )
