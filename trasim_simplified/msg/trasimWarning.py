# -*- coding = uft-8 -*-
# @Time : 2022-04-04 17:10
# @Author : yzbyx
# @File : trasimWarning.py
# @Software : PyCharm
class TrasimWarning(RuntimeWarning):
    pass


class WarningMessage:
    GAP_LESS_THAN_ZERO = '车辆{}净间距为{:.2f}m < 0\t'
    SPEED_LESS_THAN_ZERO = '车辆{}速度为{:.2f}m/s < 0\t'
    MODEL_COMPUTE_PROBLEM = '{}模型计算问题！\t'
    UNABLE_TO_PROCESS = '无法处理的特殊情况！{}\t'


if __name__ == '__main__':
    print(WarningMessage.GAP_LESS_THAN_ZERO.format('1', 10.234))
