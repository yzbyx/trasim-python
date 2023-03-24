# -*- coding = uft-8 -*-
# @Time : 2022/1/11
# @Author : yzbyx
# @File : obstacle.py
# @Software : PyCharm

from constant import V_TYPE, V_DYNAMIC, V_STATIC


class Obstacle:
    _STATIC_PARAMETER = {V_TYPE.PASSENGER: {'length': 5.0, 'width': 1.8, 'height': 1.5, 'color': [0, 255, 0],
                                            'vType': V_TYPE.PASSENGER},
                         V_TYPE.TRUCK: {'length': 7.1, 'width': 1.8, 'height': 1.5, 'color': [255, 255, 0],
                                        'vType': V_TYPE.TRUCK},
                         V_TYPE.BUS: {'length': 12.0, 'width': 1.8, 'height': 1.5, 'color': [0, 0, 255],
                                      'vType': V_TYPE.BUS}}

    _COLOR_PARAMETER = {'red': [255, 0, 0], 'black': [0, 0, 0], 'white': [255, 255, 255],
                        'blue': [0, 0, 255], 'purple': [160, 32, 240], 'yellow': [255, 255, 0],
                        'green': [0, 255, 0], 'pink': [255, 192, 203], 'gray': [190, 190, 190]}

    @classmethod
    def getStatic(cls, param1, param2):
        return cls._STATIC_PARAMETER[param1][param2]

    def __init__(self, vType=V_TYPE.PASSENGER):
        # 速度、加速度、道路ID、车道编号、偏航、角加速度、车道偏移、道路偏移
        self.dynamic = {'speed': 0, 'acc': 0, 'alpha': 0,
                        'roadId': None, 'laneNum': 0,
                        'theta': 0, 'yOffset': 0, 'xOffset': 0, 'globalPos': [0, 0]}
        self.static = self._STATIC_PARAMETER[vType]
