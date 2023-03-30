# -*- coding = uft-8 -*-
# @Time : 2022/1/11
# @Author : yzbyx
# @File : vehicle.py
# @Software : PyCharm
import warnings

import numpy as np

from trasim_simplified.kinematics.cfm import get_cf_model, CFModel
from trasim_simplified.core.constant import CFM, RUNMODE, V_TYPE, V_DYNAMIC, V_STATIC
from trasim_simplified.msg.trasimWarning import TrasimWarning, WarningMessage
from trasim_simplified.core.obstacle import Obstacle
from trasim_simplified.msg.trasimError import TrasimError, ErrorMessage


class Vehicle(Obstacle):
    _DRIVER_ID = 0
    # 由lane控制车辆的step值
    SIMULATION_STEP = 0

    def __init__(self, onLane, carType=V_TYPE.PASSENGER, mode=RUNMODE.NORMAL, fRule=CFM.IDM):
        super().__init__()
        from lane import Lane
        self.onLane: Lane = onLane
        self.isCircle = False
        self._needUpdate = True
        self.fRule: CFModel | None = None
        self.maxOffset = None
        self._isCalculate = False  # 异步更新不会使用该变量
        self._isUnderControl = False
        self._dynamicData = {}
        self.interval = None

        self.leader: Vehicle = self
        self.follower: Vehicle = self

        self._ID = 'driver' + str(Vehicle._DRIVER_ID)
        Vehicle._DRIVER_ID += 1      # 注意调用类公共属性时，一定要使用类名.属性

        self._mode = mode
        self.vehicle = Obstacle(carType)

        self.setfRule(fRule)

        self.graph = None

        self._historyOn = False  # 历史数据保存部分
        self._subscribe = []  # 历史数据订阅的内容
        self._historyData: dict[int, dict] = {}  # int为step的值，内层字典存储订阅的历史数据
        self._maxHistory = 100  # 用于控制数据量的大小

    def __eq__(self, other):
        if isinstance(other, Vehicle):
            return self.ID == other.ID
        elif isinstance(other, str):
            return self.ID == other
        else:
            raise TrasimError(ErrorMessage.OBJ_TYPE_ERROR.format(type(self), type(other)))

    def __ne__(self, other):
        if isinstance(other, Vehicle):
            return self.ID != other.ID
        elif isinstance(other, str):
            return self.ID != other
        else:
            raise TrasimError(ErrorMessage.OBJ_TYPE_ERROR.format(type(self), type(other)))

    @property
    def ID(self):
        return self._ID

    def setfRule(self, fModel=CFM.IDM):
        if isinstance(fModel, str):
            self.fRule: CFModel = get_cf_model(self, name=fModel)
        elif isinstance(fModel, CFModel):
            self.fRule: CFModel = fModel
        else:
            raise TrasimError(ErrorMessage.NO_MODEL.format(fModel))
        self.fRule.mode = self._mode

    def setfParam(self, param: dict):
        self.fRule.param_update(param)

    def setDynamic(self, param, value):
        self.vehicle.dynamic[param] = value

    def getDynamic(self, param):
        return self.vehicle.dynamic[param]

    def getStatic(self, param):
        return self.vehicle.static[param]

    def getfParam(self):
        return self.fRule.get_param_map()

    def getfStatus(self):
        return self.fRule.status

    def getHistory(self):
        return self._historyData

    def setDynamicInOperation(self, acc: float, **kwargs):
        """控制下一仿真步的加速度"""
        self._dynamicData = {'acc': acc}
        if kwargs.get('xOffset', False) and kwargs.get('speed', False):
            self._dynamicData['xOffset'] = kwargs['xOffset']
            if kwargs['xOffset'] > self.maxOffset:
                self._dynamicData['xOffset'] -= self.maxOffset
            self._dynamicData['speed'] = kwargs['speed']
            self._needUpdate = False
        self._isUnderControl = True

    def historyOn(self, on=True, subscribe: list = None, maxHistory=np.inf):
        """开启运行状态历史记录"""
        self._historyOn = on
        if not on:
            return
        sTemp = []
        if subscribe is not None:
            for s in subscribe:
                if s not in self.vehicle.dynamic.keys():
                    message = f"{s}不在可订阅的动态属性列表中！"
                    warnings.warn(message, RuntimeWarning)
                else:
                    sTemp.append(s)
        else:
            sTemp = list(self.vehicle.dynamic.keys())
        self._subscribe = sTemp
        self._maxHistory = maxHistory

    def _underControlUpdate(self):
        """被控制车辆的数据更新方法"""
        if self._needUpdate:
            acc = self._dynamicData['acc']
            v = self.getDynamic('speed')
            x = self.getDynamic('xOffset')
            if v + acc * self.interval < 0:
                acc = - v / self.interval
            speed = acc * self.interval + v
            self._dynamicData['speed'] = speed
            xOffset = x + speed * self.interval + 0.5 * acc * (self.interval ** 2)
            self._dynamicData['xOffset'] = (xOffset if xOffset <= self.maxOffset else xOffset - self.maxOffset)
        else:
            self._needUpdate = True

        self.vehicle.dynamic.update(self._dynamicData)
        self._isUnderControl = False
        self._isCalculate = False

    def update(self, interval=0.1, updateMethod='synchronous'):
        """供外部调用的车辆更新方法，含有tau参数的模型interval始终为tau"""
        # 数据计算
        # tau = self.fRule.getTau()
        # if tau != 0:
        #     interval = tau
        self.interval = interval
        if updateMethod == 'synchronous':
            if self._isCalculate is False:
                if self._isUnderControl is False:
                    self._followRule(updateMethod)
                else:
                    self._isCalculate = True
            else:
                if not self._isUnderControl:
                    self._circle_xOffset_reset()
                    self.vehicle.dynamic.update(self._dynamicData)
                    self._isCalculate = False
                else:
                    self._underControlUpdate()
                self._dataStorage()
        elif updateMethod == 'asynchronous':
            if self._isUnderControl is False:
                self._followRule(updateMethod)
            else:
                self._underControlUpdate()
            self._dataStorage()
        else:
            raise ValueError(f"updateMethod={updateMethod}")

    def _dataStorage(self):
        #  数据存储
        if self._historyOn and len(self._subscribe) > 0:
            subscription = {}
            for s in self._subscribe:
                subscription[s] = self.vehicle.dynamic[s]
            self._historyData[self.SIMULATION_STEP] = subscription
            if len(self._historyData) > self._maxHistory:
                historyTimeList = list(self._historyData.keys())
                delTime = historyTimeList[0]
                self._historyData.pop(delTime)

    def _followRule(self, updateMethod):
        """车辆跟驰行为计算"""
        result = self.fRule.step()

        pre_speed = self.vehicle.dynamic["speed"]
        tau = self.fRule.getTau()
        next_speed = pre_speed + result * tau

        self._dynamicData = {"acc": result,
                             "speed": next_speed,
                             "xOffset": self.vehicle.dynamic["xOffset"] + (pre_speed + next_speed) / 2 * tau}

        if updateMethod == 'synchronous':
            self._isCalculate = True
        elif updateMethod == 'asynchronous':
            self._circle_xOffset_reset()
            self.vehicle.dynamic.update(self._dynamicData)

    def _circle_xOffset_reset(self):
        self.isCircle = False
        x = self._dynamicData[V_DYNAMIC.X_OFFSET]
        if x >= self.maxOffset:
            x -= self.maxOffset
            self.onLane.first = self.follower
            self.onLane.last = self
            self.isCircle = True
            self._dynamicData[V_DYNAMIC.X_OFFSET] = x

    def selfCheck(self):
        xOffset = self.vehicle.dynamic[V_DYNAMIC.X_OFFSET]
        leaderX = self.leader.vehicle.dynamic[V_DYNAMIC.X_OFFSET]
        leaderL = self.leader.vehicle.static[V_STATIC.LENGTH]
        if leaderX - leaderL < xOffset:
            TrasimWarning(WarningMessage.GAP_LESS_THAN_ZERO.format(self._ID, leaderX - leaderL - xOffset))
