# -*- coding = uft-8 -*-
# @Time : 2022/1/11
# @Author : yzbyx
# @File : obstacle.py
# @Software : PyCharm

from trasim_simplified.core.constant import V_TYPE


class Obstacle:
    _STATIC_PARAMETER = {V_TYPE.PASSENGER: {'length': 5.0, 'width': 1.8, 'height': 1.5, 'color_value': [0, 255, 0],
                                            'vType': V_TYPE.PASSENGER},
                         V_TYPE.TRUCK: {'length': 7.1, 'width': 1.8, 'height': 1.5, 'color_value': [255, 255, 0],
                                        'vType': V_TYPE.TRUCK},
                         V_TYPE.BUS: {'length': 12.0, 'width': 1.8, 'height': 1.5, 'color_value': [0, 0, 255],
                                      'vType': V_TYPE.BUS}}

    _COLOR_PARAMETER = {'red': [255, 0, 0], 'black': [0, 0, 0], 'white': [255, 255, 255],
                        'blue': [0, 0, 255], 'purple': [160, 32, 240], 'yellow': [255, 255, 0],
                        'green': [0, 255, 0], 'pink': [255, 192, 203], 'gray': [190, 190, 190]}

    @classmethod
    def getStatic(cls, param1, param2):
        return cls._STATIC_PARAMETER[param1][param2]

    def __init__(self, type_: str):
        self.x = 0
        self.v = 0
        self.a = 0
        self.color = self._COLOR_PARAMETER["yellow"]

        self.length = 5.0
        self.width = 1.8
        self.type = type_
