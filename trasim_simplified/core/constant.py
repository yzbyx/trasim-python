# -*- coding = uft-8 -*-
# @time : 2022-03-09 16:37
# @Author : yzbyx
# @File : constant.py
# @Software : PyCharm
# simulation-simplified常量集合
# ******************************
# 随机种子
# ******************************
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Callable

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from trasim_simplified.core.agent import Vehicle
    from trasim_simplified.core.agent.game_agent import Game_Vehicle
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract


class RANDOM_SEED:
    CFM_SEED = 0  # 用于跟驰模型
    LCM_SEED = 0  # 用于换道模型
    CAR_PRODUCE_SEED = 0  # 用于车辆生成


class TrackInfo:
    # ------Vehicle Tracks------ #
    # basic 单车道仿真
    frame = "frame"
    """当前帧ID"""
    time = "time"
    """当前帧时间 [s]"""
    trackId = "trackId"
    """车辆ID"""
    Local_X = "localLon"
    """车头中点投影到车道线的车道起点偏移量 [m]"""
    Local_Y = "localLat"
    """车头中点距离所在车道中轴线的偏移量 [m]"""
    length = "length"
    """车辆长度 [m]"""
    v_Class = "v_Class"
    """车辆类别 [Car/Truck]"""
    Lane_ID = "laneId"
    """车道ID [0/1/2...]"""

    # extend basic-1 车辆基础信息拓展
    acc = "acc"
    """车辆的加速度 [m/s^2]"""
    speed = "speed"
    """车辆的速度 [m/s]"""
    heading = "heading"
    """沿车道方向的车辆方向夹角（逆时针为正） [radian]"""
    localLonVel = "localLonVel"
    """车辆平行于车道中轴线的速度分量 [m/s]"""
    localLatVel = "localLatVel"
    """车辆垂直于车道中轴线的速度分量 [m/s]"""
    localLonAcc = "localLonAcc"
    """车辆平行于车道中轴线的加速度分量 [m/s^2]"""
    localLatAcc = "localLatAcc"
    """车辆垂直于车道中轴线的加速度分量 [m/s^2]"""
    delta = "delta"
    """车辆前轮转向角 [radian]"""

    # extended basic-2 车辆关系拓展
    Following_ID = "Following_ID"
    """后车ID"""
    Preceding_ID = "Preceding_ID"
    """前车ID"""
    Left_Preceding_ID = "Left_Preceding_ID"
    """左前车ID（车头基准）"""
    Left_Following_ID = "Left_Following_ID"
    """左后车ID（车头基准）"""
    Right_Preceding_ID = "Right_Preceding_ID"
    """右前车ID（车头基准）"""
    Right_Following_ID = "Right_Following_ID"
    """右后车ID（车头基准）"""
    dv = "dv"
    """当前车与前车的差值 [m/s]"""
    gap = "gap"
    """与前车的净间距 [m]"""
    dhw = "dhw"
    """与前车的车头间距 [m]"""
    thw = "thw"
    """与前车的车头时距 [s]"""
    ttc = "ttc"
    """与前车的碰撞时间 [s]"""

    # multi lane 多车道单方向
    width = "width"
    """车辆宽度"""

    # level-1 extended 碰撞箱详细表述
    yaw = "yaw"
    """车辆全局朝向 [radian]"""

    # extended level-3 坐标详细表述
    yFrontGlobal = "yFrontGlobal"
    """车头中点的全局y坐标，横轴为x，纵轴为y [m]"""
    xFrontGlobal = "xFrontGlobal"
    """车头中点的全局x坐标，横轴为x，纵轴为y [m]"""
    xCenterGlobal = "xCenterGlobal"
    yCenterGlobal = "yCenterGlobal"

    Pair_ID = "pairId"

    # 兼容trasim_simplified
    lane_add_num = Lane_ID
    id = trackId
    step = frame
    v = localLonVel
    x = Local_X
    y = Local_Y
    a = localLonAcc

    cf_acc = "cf_acc"
    cf_id = "cf model ID"
    """跟驰模型类别ID"""
    lc_id = "lc model ID"
    """换到模型类别ID"""
    car_type = v_Class
    """车辆类型"""

    safe_ttc = "ttc (s)"
    safe_tet = "tet"
    safe_tit = "tit (s)"
    safe_picud = "picud (m)"
    safe_picud_KK = "picud_KK (m)"


class Prefix:
    leader = "leader_"
    follower = "follower_"


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
    IDM_VZ = "IDM_VZ"
    IDM_VS = "IDM_VS"
    IDM_SZ = "IDM_SZ"
    CTM = "CTM"
    """CTM系列"""
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
    """"""
    ACC = "ACC/TPACC"
    """Kerner对自动驾驶类车辆的换道规则"""
    MOBIL = "MOBIL"
    """MOBIL换道模型"""
    APF = "APF"
    """基于势场的换道模型"""


class SECTION_TYPE(Enum):
    BASE = 0
    """基本路段"""
    ON_RAMP = 1
    """入口匝道区域"""
    OFF_RAMP = 2
    """出口匝道区域"""
    AUXILIARY = 3
    """辅助车道区域"""


class MARKING_TYPE(Enum):
    SOLID = 1
    DASHED = 2


class V_CLASS(Enum):
    BASE = 0
    """基本车辆"""
    GAME_HV = 1
    """博弈人类车辆"""
    GAME_AV = 2
    """博弈自动驾驶车辆"""


# ******************************
# 车辆类别
# ******************************
class V_TYPE(Enum):
    # 汽车
    PASSENGER = 0
    # 货车
    TRUCK = 1
    # 公交车
    BUS = 2
    OBSTACLE = -1

    @classmethod
    def get_all_v_type_no_obstacle(cls):
        dict_ = V_TYPE.__dict__
        values = {}
        for key in dict_.keys():
            if isinstance(dict_[key], int) and key[:2] != "__" and key != V_TYPE.OBSTACLE:
                values.update({key: dict_[key]})
        return values


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


class LaneMarkingType(Enum):
    """
    与carla.LaneMarkingType对应
    """
    NONE = 0
    BottsDots = 1
    Broken = 2  # 虚线
    BrokenBroken = 3
    BrokenSolid = 4
    Curb = 5
    Grass = 6
    Solid = 7  # 实线
    SolidBroken = 8
    SolidSolid = 9
    Other = 10


class LaneType(Enum):
    """
    与carla.LaneType对应
    """
    NONE = 0
    Driving = 1
    Sidewalk = 2
    Shoulder = 3
    Parking = 4
    Curb = 5
    Grass = 6
    Other = 7


class LaneMarkingColor(Enum):
    """
    与carla.LaneMarkingColor对应
    """
    NONE = 0
    Yellow = 1
    White = 2
    Blue = 3
    Red = 4
    Green = 5
    Other = 6


class LaneChange(Enum):
    """
    与carla.LaneChange对应
    """
    NONE = 0
    Left = 1
    Right = 2
    Both = 3
    Other = 4


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


class RouteType(Enum):
    diverge = 0
    merge = 1
    mainline = 2
    auxiliary = 3


@dataclass
class PyGameConfig:
    verbose = False
    """是否输出详细信息"""
    res = '1280x720'
    """窗口分辨率"""


@dataclass
class VehSurr:
    ev: 'Vehicle' = None
    cp: 'Vehicle' = None
    cr: 'Vehicle' = None
    lp: 'Vehicle' = None
    lr: 'Vehicle' = None
    rp: 'Vehicle' = None
    rr: 'Vehicle' = None


@dataclass
class TrajPoint:
    x: float = None
    y: float = None
    yaw: float = None
    speed: float = None
    acc: float = None
    delta: float = None
    length: float = None
    width: float = None

    @property
    def vx(self):
        return self.speed * np.cos(self.yaw)

    @property
    def vy(self):
        return self.speed * np.sin(self.yaw)

    @property
    def ax(self):
        return self.acc * np.cos(self.yaw)

    @property
    def ay(self):
        return self.acc * np.sin(self.yaw)

    def to_ndarray(self):
        """x, vx, ax, y, vy, ay"""
        return np.array([self.x, self.vx, self.ax, self.y, self.vy, self.ay])

    def copy(self):
        return TrajPoint(self.x, self.y, self.yaw, self.speed, self.acc, self.delta, self.length, self.width)

    def to_center(self):
        """将坐标转换为车辆中心坐标系"""
        return np.array([
            self.x - self.length / 2 * np.cos(self.yaw),
            self.vx, self.ax,
            self.y - (self.width / 2) * np.sin(self.yaw),
            self.vy, self.ay,
            np.cos(self.yaw), np.sin(self.yaw), self.length, self.width
        ])


@dataclass
class TrajData:
    EV_traj: Optional[np.ndarray]
    """换道车辆的轨迹"""
    TF_traj: Optional[np.ndarray]
    """目标间隙前车的轨迹"""
    TR_traj: Optional[np.ndarray]
    """目标间隙后车的轨迹"""
    PC_traj: Optional[np.ndarray]
    """当前车道前车的轨迹"""
    TF_PC_traj: Optional[np.ndarray]
    """目标间隙前车的前车轨迹"""
    TR_PC_traj: Optional[np.ndarray]
    """目标间隙后车的前车轨迹"""
    PC_PC_traj: Optional[np.ndarray]
    """前车的前车轨迹"""
    TF_cost_lambda: Optional[Callable]
    TR_cost_lambda: Optional[Callable]


@dataclass
class GameRes:
    cost_df: Optional[pd.DataFrame]
    """成本函数"""
    TF: 'Game_Vehicle'
    """目标间隙前车"""
    TR: 'Game_Vehicle'
    """目标间隙后车"""
    PC: 'Game_Vehicle'
    """当前车道前车"""
    EV_stra: Optional[float]
    """换道车辆的最优换道时间"""
    TF_stra: Optional[float]
    """目标间隙前车的期望时距"""
    TR_real_stra: Optional[float]
    """目标间隙后车的期望时距"""
    EV_cost: Optional[float]
    """换道车辆的成本函数"""
    TF_cost: Optional[float]
    """目标间隙前车的成本函数"""
    TR_real_cost: Optional[float]
    """目标间隙后车的成本函数"""
    EV_opti_series: Optional[pd.Series]
    """换道车辆的最优时距下的成本函数序列"""
    traj_data: Optional[TrajData] = None
    """其他车辆的轨迹"""


@dataclass
class GapJudge:
    lc_direction: float
    target_acc: Optional[float]
    """速度调整的加速度"""
    adapt_time: float
    """调整时间"""
    EV: 'Game_Vehicle'
    """换道车辆"""
    TF: Optional['Game_Vehicle'] = None
    """目标间隙前车"""
    TR: Optional['Game_Vehicle'] = None
    """目标间隙后车"""
    PC: Optional['Game_Vehicle'] = None
    """当前车道前车"""
    acc_gain: Optional[float] = None
    """加速度增益"""
    ttc_risk: Optional[float] = None
    """碰撞风险"""
    route_gain: Optional[float] = None
    """路径成本函数"""
    adapt_end_time: float = None
    """调整结束时间"""
    target_lane: Optional['LaneAbstract'] = None
    """目标车道"""


@dataclass
class ScenarioTraj:
    dataset_name: str
    """数据集名称"""
    pattern_name: str
    """场景名称"""
    track_id: int
    """车辆ID"""
    lc_start_frame: int
    lc_frame: int
    lc_end_frame: int

    EV_traj: Optional[pd.DataFrame]

    TP_traj: Optional[pd.DataFrame] = None
    TR_traj: Optional[pd.DataFrame] = None

    CP_traj: Optional[pd.DataFrame] = None
    CPP_traj: Optional[pd.DataFrame] = None

    TRR_traj: Optional[pd.DataFrame] = None
    TPP_traj: Optional[pd.DataFrame] = None

    OR_traj: Optional[pd.DataFrame] = None
    ORR_traj: Optional[pd.DataFrame] = None

    OP_traj: Optional[pd.DataFrame] = None
    OPP_traj: Optional[pd.DataFrame] = None


class ScenarioMode(Enum):
    NO_INTERACTION = 0
    """无交互"""
    INTERACTION = 1
    """有交互"""


if __name__ == '__main__':
    print(dir(V_STATIC))
