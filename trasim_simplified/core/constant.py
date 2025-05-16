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

from trasim_simplified.core.agent.utils import get_xy_quintic

if TYPE_CHECKING:
    from trasim_simplified.core.agent import Vehicle
    from trasim_simplified.core.agent.game_agent import Game_Vehicle
    from trasim_simplified.core.agent import Game_A_Vehicle
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
    localLon = "localLon"
    """车头中点投影到车道线的车道起点偏移量 [m]"""
    localLat = "localLat"
    """车头中点距离所在车道中轴线的偏移量 [m]"""
    length = "length"
    """车辆长度 [m]"""
    v_Class = "v_Class"
    """车辆类别 [Car/Truck]"""
    laneId = "laneId"
    """车道ID [0/1/2...]"""

    # extend basic-1 车辆基础信息拓展
    acc = "acceleration"
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
    roadLon = "roadLon"
    roadLat = "roadLat"

    isLC = "isLC"
    gap_res_list = "gap_res_list"
    """换道决策结果列表"""
    opti_gap_res = "gap_res"
    """换道决策结果"""
    game_res_list = "game_res_list"
    """博弈决策结果列表"""
    opti_game_res = "game_res"
    """博弈决策结果"""

    Pair_ID = "pairId"

    # 兼容trasim_simplified
    lane_add_num = laneId
    id = trackId
    step = frame
    v = localLonVel
    x = localLon
    y = localLat
    a = localLonAcc

    cf_acc = "cf_acc"
    cf_id = "cf model ID"
    """跟驰模型类别ID"""
    lc_id = "lc model ID"
    """换到模型类别ID"""
    car_type = v_Class
    """车辆类型"""

    tet = "tet"
    tit = "tit (s)"
    picud = "picud (m)"
    picud_KK = "picud_KK (m)"


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
    GAME_HV = "HV"
    """博弈人类车辆"""
    GAME_AV = "AV"
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
    CR_traj: Optional[np.ndarray]
    """当前车道后车的轨迹"""
    TF_cost_lambda: Optional[Callable]
    TR_cost_lambda: Optional[Callable]
    CP_cost_lambda: Optional[Callable]
    CR_cost_lambda: Optional[Callable]


@dataclass
class GameVehSurr:
    lc_direction: int
    EV: 'Game_A_Vehicle'
    TR: 'Game_Vehicle'
    TP: 'Game_Vehicle'
    CR: 'Game_Vehicle'
    CP: 'Game_Vehicle'

    TRR: 'Game_Vehicle' = None
    TPP: 'Game_Vehicle' = None
    CRR: 'Game_Vehicle' = None
    CPP: 'Game_Vehicle' = None

    def clone(self):
        return GameVehSurr(
            self.lc_direction,
            self.EV,
            self.TR,
            self.TP,
            self.CR,
            self.CP,
            self.TRR,
            self.TPP,
            self.CRR,
            self.CPP
        )


@dataclass
class LcGap:
    TP: "Vehicle"
    TR: "Vehicle"
    CP: "Vehicle"

    def __eq__(self, other):
        if not isinstance(other, LcGap):
            return False
        return (self.TP.ID == other.TP.ID and
                self.TR.ID == other.TR.ID and
                self.CP.ID == other.CP.ID)

    def __repr__(self):
        return f"TP: {self.TP.ID}, TR: {self.TR.ID}, CP: {self.CP.ID}"


@dataclass
class StraInfo:
    veh: "Vehicle"
    stra_time: Optional[float]
    cf_stra: float
    lc_direction: int
    lc_gap: Optional[LcGap] = None
    solve_res: Optional['SolveRes'] = None

    def __eq__(self, other):
        if not isinstance(other, StraInfo):
            return False
        return (self.veh.ID == other.veh.ID and
                self.stra_time == other.stra_time and
                self.cf_stra == other.cf_stra and
                self.lc_direction == other.lc_direction and
                self.lc_gap == other.lc_gap)

    def __lt__(self, other):
        if not isinstance(other, StraInfo):
            return False
        return hash(self) < hash(other)

    @property
    def lane(self):
        return self.veh.lane

    @property
    def target_lane(self):
        if self.lc_direction == 1:
            return self.lane.right_neighbour_lane
        elif self.lc_direction == -1:
            return self.lane.left_neighbour_lane
        else:
            return self.lane

    def copy(self):
        return StraInfo(self.veh, self.stra_time, self.cf_stra, self.lc_direction, self.lc_gap)

    def __hash__(self):
        if self.lc_gap is None:
            return hash((self.veh.ID, float(self.stra_time), float(self.cf_stra), self.lc_direction))
        return hash((self.veh.ID, float(self.stra_time), float(self.cf_stra), self.lc_direction,
                     self.lc_gap.TP.ID, self.lc_gap.TR.ID, self.lc_gap.CP.ID))

    def __repr__(self):
        return f"veh: {self.veh.ID}, stra_time: {self.stra_time:.3f}, cf_stra: {self.cf_stra:.3f}, " \
               f"lc_direction: {self.lc_direction}, lc_gap: {self.lc_gap}, total_cost: {self.solve_res.cost:.3f}, "\
               f"safe: {self.solve_res.safe:.3f}, com: {self.solve_res.com:.3f}, " \
               f"eff: {self.solve_res.eff:.3f}, route: {self.solve_res.route:.3f}"


@dataclass
class GameRes:
    step: int
    """当前决策时步"""
    game_surr: Optional[GameVehSurr]
    total_cost: Optional[float]

    EV_stra: Optional[StraInfo]
    TF_stra: Optional[StraInfo]
    TR_stra: Optional[StraInfo]
    CR_stra: Optional[StraInfo]
    CP_stra: Optional[StraInfo]

    EV_cost: Optional[float]
    TP_cost: Optional[float]
    TR_cost: Optional[float]
    CR_cost: Optional[float]
    CP_cost: Optional[float]

    traj_data: Optional[TrajData] = None
    """其他车辆的轨迹"""
    EV_opti_traj: Optional[np.ndarray] = None
    EV_lc_step: Optional[int] = None

    TR_esti_lambda: Optional[Callable] = None
    TR_real_lambda: Optional[Callable] = None
    CR_esti_lambda: Optional[Callable] = None
    CR_real_lambda: Optional[Callable] = None

    def __repr__(self):
        return f"step: {self.step}, " \
               f"EV: {self.game_surr.EV.ID}-{self.game_surr.EV.NAME}, "\
               f"TP: {self.game_surr.TP.ID}-{self.game_surr.TP.NAME}, "\
               f"TR: {self.game_surr.TR.ID}-{self.game_surr.TR.NAME}, "\
               f"CP: {self.game_surr.CR.ID}-{self.game_surr.CR.NAME}, "\
               f"CR: {self.game_surr.CP.ID}-{self.game_surr.CP.NAME}\n"\
               f"EV: {self.EV_stra}\nTP: {self.TF_stra}\nTR: {self.TR_stra}\nCR: {self.CR_stra}\nCP: {self.CP_stra}"


@dataclass
class SolveRes:
    quintic: Optional[np.ndarray]
    times: Optional[np.ndarray]
    safe: float
    com: float
    eff: float
    route: float
    cost: float

    cost_lambda: Optional[Callable] = None
    _traj: Optional[np.ndarray] = None

    @property
    def traj(self):
        if self._traj is None:
            self._traj = np.vstack([np.array(get_xy_quintic(self.quintic, t)) for t in self.times])
        return self._traj

    def set_traj(self, traj):
        self._traj = traj


@dataclass
class GapJudge:
    step: int
    lc_direction: float
    gap: Optional[float]
    target_acc: Optional[float]
    """速度调整的加速度"""
    adapt_time: float
    """调整时间"""
    EV: 'Vehicle'
    """换道车辆"""
    TF: Optional['Vehicle'] = None
    """目标间隙前车"""
    TR: Optional['Vehicle'] = None
    """目标间隙后车"""
    PC: Optional['Vehicle'] = None
    """当前车道前车"""
    acc_gain: Optional[float] = None
    """加速度增益"""
    ttc_risk: Optional[float] = None
    """碰撞风险"""
    route_gain: Optional[float] = None
    """目标路径增益"""
    platoon_gain: Optional[float] = None
    """队列换道增益"""
    weaving_gain: Optional[float] = None
    """交织换道增益"""
    adapt_end_time: float = None
    """调整结束时间"""
    target_lane: Optional['LaneAbstract'] = None
    """目标车道"""
    lc_prob: Optional[float] = -1
    """换道概率"""

    def __repr__(self):
        return f"step: {self.step}, EV: {self.EV.ID}, lc_d: {self.lc_direction}, tar_a: {self.target_acc}, " \
               f"a_time: {self.adapt_time}, gap: {self.gap}, "\
               f"a_gain: {self.acc_gain:.3f}, ttc_risk: {self.ttc_risk:.3f}, route_gain: {self.route_gain:.3f} " \
               f"lc_prob: {self.lc_prob:.3f}, "\
               f"TF: {self.TF.ID}, TR: {self.TR.ID}, PC: {self.PC.ID}, "


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

    CR_traj: Optional[pd.DataFrame] = None
    CRR_traj: Optional[pd.DataFrame] = None


class ScenarioMode:
    NO_INTERACTION = "无交互"
    """第一类（无交互）"""
    INTERACTION_TR_HV_TP_HV = "有交互(TR-HV)"
    """第二类（有交互TR_HV_TP_HV）"""
    INTERACTION_TR_AV_TP_HV = "有交互(TR-AV)"
    """第三类（有交互，TR_AV_TP_HV）"""
    INTERACTION_TR_HV_TP_AV = "有交互(TR-HV TP-AV)"
    """第三类（有交互，TR_HV_TP_AV）"""
    INTERACTION_TR_AV_TP_AV = "有交互(TR-AV TP-AV)"
    """第三类（有交互，TR_AV_TP_AV）"""


if __name__ == '__main__':
    print(dir(V_STATIC))
