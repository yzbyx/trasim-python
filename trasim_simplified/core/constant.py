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
    LCM_SEED = 0  # 用于换道模型
    CAR_PRODUCE_SEED = 0  # 用于车辆生成


class COLOR:
    red = [255, 0, 0]
    black = [0, 0, 0]
    white = [255, 255, 255]
    blue = [0, 0, 255]
    purple = [160, 32, 240]
    yellow = [255, 255, 0]
    green = [0, 255, 0]
    pink = [255, 192, 203]
    gray = [190, 190, 190]


# ******************************
# 跟驰模型
# ******************************
class CFM:
    LCM = "Longitude_control_model"
    "Ni的纵向控制模型"
    CACC = "CACC"
    """CACC模型"""
    DUMMY = "Dummy"
    """虚拟车辆对应模型"""
    IDM = 'IDM'
    """IDM模型"""
    GIPPS = 'Gipps'
    """Gipps模型"""
    WIEDEMANN_99 = 'W99'
    """Wiedemann99模型"""
    NON_LINEAR_GHR = 'Non_linear_GHR'
    """GM模型"""
    OPTIMAL_VELOCITY = 'Optimal_Velocity'
    """OV模型"""
    KK = 'Three-Phase_KK'
    """KK模型"""
    LINEAR = "Linear"
    """线性跟驰模型(刺激反应模型)"""
    TPACC = "Three-Phase_TPACC"
    """TPACC模型"""
    ACC = "ACC"
    """ACC模型"""


class LCM:
    KK = 'KK'
    """KK模型的换道策略"""
    ACC = "ACC/TPACC"
    """Kerner对自动驾驶类车辆的换道规则"""


class SECTION_TYPE:
    BASE = "base"
    """基本路段"""
    ON_RAMP = "on_ramp"
    """入口匝道区域"""
    OFF_RAMP = "off_ramp"
    """出口匝道区域"""
    NO_LEFT = "no_lc_to_left"
    """禁止向左换道"""
    NO_RIGHT = "no_lc_to_right"
    """禁止向右换道"""


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
    OBSTACLE = "obstacle"


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
