# -*- coding = uft-8 -*-
# @Time : 2022/1/11
# @Author : yzbyx
# @File : obstacle.py
# @Software : PyCharm
from trasim_simplified.core.constant import COLOR


class Obstacle:
    def __init__(self, type_: int):
        self.x = 0
        self.v = 0
        self.a = 0
        self.color = COLOR.yellow

        self.length = 5.0
        self.width = 1.8
        self.type = type_
