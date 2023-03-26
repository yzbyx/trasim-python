# -*- coding = uft-8 -*-
# @Time : 2023-03-25 23:06
# @Author : yzbyx
# @File : test.py
# @Software : PyCharm

class A:
    def __init__(self):
        self._init()

    def _init(self):
        self.i = 0

class B(A):
    def __init__(self):
        super().__init__()

    def _init(self):
        self.i = 1

b = B()
print(b.i)
