# -*- coding = uft-8 -*-
# @Time : 2022-03-09 16:37
# @Author : yzbyx
# @File : constant.py
# @Software : PyCharm
# simulation-simplified常量集合
# ******************************
# 随机种子
# ******************************
class RANDOM_SEED:
    CFM_SEED = 0  # 用于跟驰模型
    CAR_PRODUCE_SEED = 0  # 用于车辆生成


# ******************************
# 跟驰模型
# ******************************
class CFM:
    # idm模型
    IDM = 'IDM'
    # gipps模型
    GIPPS = 'Gipps'
    # wiedemann模型
    WIEDEMANN_99 = 'W99'
    # GM模型
    NON_LINEAR_GHR = 'Non_linear_GHR'
    # OV模型
    OPTIMAL_VELOCITY = 'Optimal_Velocity'
    # TPACC模型
    KK = 'Three-Phase_ACC'
    LINEAR = "Linear"
    """线性跟驰模型(刺激反应模型)"""


# ******************************
# 车辆类别
# ******************************
class V_TYPE:
    # 汽车
    PASSENGER = 'passenger'
    # 货车
    TRUCK = 'truck'
    # 公交车
    BUS = 'bus'


# ******************************
# 车辆静态属性
# ******************************
class V_STATIC:
    # 车身长度
    LENGTH = 'length'
    # 车身宽度
    WIDTH = 'width'
    # 车身高度
    HEIGHT = 'height'
    # 车辆类型
    TYPE = 'vType'


# ******************************
# 车辆动态属性
# ******************************
class V_DYNAMIC:
    # 车辆偏移
    X_OFFSET = 'xOffset'
    # 车辆速度
    VELOCITY = 'speed'
    # 车辆加速度
    ACC = 'acc'


# ******************************
# 仿真模式
# ******************************
class RUNMODE:
    # 带警告输出的普通模式
    NORMAL = 'normal'
    # 无警告输出的安静模式
    SILENT = 'silent'


if __name__ == '__main__':
    print(dir(V_STATIC))
