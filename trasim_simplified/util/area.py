# -*- coding = uft-8 -*-
# @Time : 2022-05-03 22:52
# @Author : yzbyx
# @File : area.py
# @Software : PyCharm
# Modified from XiaoYW: https://blog.csdn.net/xiaoyw71/article/details/79952520
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _getAreaByVector(points: list[Point]):
    area = 0
    if len(points) < 3:
        raise RuntimeError('点的数量需要大于等于3！')
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        # TODO: 需要理解
        try:
            triArea = (p1.x * p2.y - p2.x * p1.y) / 2
            area += triArea
        except TypeError as e:
            print(e)
            print(f'Type: {type(p1.x)}\tValue: {p1.x}')
            print(f'Type: {type(p1.y)}\tValue: {p1.y}')
            print(f'Type: {type(p2.x)}\tValue: {p2.x}')
            print(f'Type: {type(p2.y)}\tValue: {p2.y}')
    return area


def getAreaByOrderedPoints(x, y):
    """逆时针数据为正，顺时针为负"""
    if len(x) != len(y):
        raise RuntimeError(f'x的长度{len(x)}!=y的长度{len(y)}')
    if len(x) < 3:
        raise RuntimeError(f'数据长度小于3！')
    points = []
    for i in range(len(x)):
        points.append(Point(x[i], y[i]))
    points.append(points[0])

    area = _getAreaByVector(points)
    return area


def get_hysteresis_by_integral(speed: np.ndarray, spacing: np.ndarray):
    """
    要首尾相连，即最后一个点的位置等于第一个点

    :_param speed: 点序列的x坐标
    :_param spacing: 点序列的y坐标
    :return: 返回点列围合平面的平均y差值，顺时针为正，逆时针为负
    """
    assert (speed[0] == speed[-1]) and (spacing[0] == spacing[-1]), "要首尾相连，即最后一个点的位置等于第一个点"
    # 计算下一个点的与当前点的x和y的变化量
    diff_speed = np.diff(speed)
    diff_spacing = np.diff(spacing)
    spacing = spacing[:-1] + diff_spacing / 2  # 计算前后两点的平均y值
    speed_range = np.max(speed) - np.min(speed)
    return np.sum(diff_speed * spacing) / speed_range  # 求和积分


def _test():
    x = [1, 2, 2, 1]
    y = [1, 1, 2, 2]
    print(getAreaByOrderedPoints(x, y))


if __name__ == '__main__':
    _test()
