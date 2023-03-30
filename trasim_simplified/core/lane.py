# -*- coding = uft-8 -*-
# @Time : 2022/1/11
# @Author : yzbyx
# @File : lane.py
# @Software : PyCharm
import random
from typing import Sequence

import numpy as np

from trasim_simplified.core.vehicle import Vehicle
from trasim_simplified.core.obstacle import Obstacle
from trasim_simplified.core.constant import CFM, RUNMODE, V_TYPE, V_STATIC, V_DYNAMIC, RANDOM_SEED
from trasim_simplified.msg.trasimError import TrasimError, ErrorMessage as vem
from trasim_simplified.msg.trasimWarning import TrasimWarning, WarningMessage as wm


class Lane:
    # _NO_TAU = [CFM.IDM, CFM.WIEDEMANN_99]
    # _NO_TEST_COLLUSION = [CFM.NON_LINEAR_GHR]
    _SIMULATION_STEP = 0
    _LANE_ID = 0

    def __init__(self, length=1000, operateMethod='circle', mode=RUNMODE.NORMAL):
        self.count = {}
        Lane._LANE_ID += 1  # 注意调用类公共属性时，一定要使用类名.属性
        self._ID = 'lane' + str(Lane._LANE_ID)
        self.length = length
        self.rotate = 0.0
        self.globalPos = [0, 0]  # 车道全局起点位置，车道起点截面中点
        self._method = operateMethod
        self._mode = mode

        self._driverList: list[Vehicle] = []
        self.first: Vehicle | None = None
        self.last: Vehicle | None = None

        self.car_produce_seed = RANDOM_SEED.CAR_PRODUCE_SEED

    @property
    def ID(self):
        return self._ID

    @property
    def SIMULATION_STEP(self):
        return self._SIMULATION_STEP

    def updateDriverRelation(self):
        """车辆纵向关系排列，在随机放置大量车辆后调用"""
        self._sortDriverList()
        if len(self._driverList) == 1:
            driver = self._driverList[0]
            driver.leader = driver.follower = driver
            self.last = self.first = driver
        else:
            for i, driver in enumerate(self._driverList):
                if i == 0:
                    self.last = driver
                    driver.leader = self._driverList[i + 1]
                    driver.follower = self._driverList[-1]
                elif i == len(self._driverList) - 1:
                    self.first = driver
                    driver.follower = self._driverList[i - 1]
                    driver.leader = self._driverList[0]
                else:
                    driver.leader = self._driverList[i + 1]
                    driver.follower = self._driverList[i - 1]
        self._relationCheck()

    def _sortDriverList(self):
        self._driverList.sort(key=lambda x: x.vehicle.dynamic[V_DYNAMIC.X_OFFSET])

    def _relationCheck(self):
        """被动检查超车重叠问题"""
        start = self.last
        now = start
        leader = start.leader
        isEnd = False
        while not isEnd:
            if leader == start:
                isEnd = True
            leaderX = leader.getDynamic(V_DYNAMIC.X_OFFSET)
            nowX = now.getDynamic(V_DYNAMIC.X_OFFSET)
            leaderL = leader.getStatic(V_STATIC.LENGTH)
            if leaderX - leaderL < nowX:
                TrasimWarning(wm.GAP_LESS_THAN_ZERO.format(now.ID, leaderX - leaderL - nowX))
                if leaderX < nowX:
                    TrasimError(vem.HEADWAY_LESS_THAN_ZERO.format(now.ID, leaderX - nowX))
            now = leader
            leader = leader.leader

    def delDriver(self, driver: str | Sequence[str] | Vehicle | Sequence[Vehicle]):
        if isinstance(driver, (str, Vehicle)):
            self._delSingleDriver(driver)
        elif isinstance(driver, Sequence):
            if len(driver) != 0:
                for d in driver:
                    self._delSingleDriver(d)

    def _delSingleDriver(self, _driver: str | Vehicle):
        """删除对应车辆"""
        driver, index = self._findDriver(_driver)
        self._driverList.pop(index)
        # 清除车辆关系
        follower = driver.follower
        leader = driver.leader
        follower.leader = leader
        leader.follower = follower

    def _findDriver(self, driver: str | Vehicle) -> tuple[Vehicle, int]:
        index = self._driverList.index(driver)
        return self._driverList[index], index

    def addDriverTool(self, carNum: int, carType=V_TYPE.PASSENGER, method: str = 'uniform',
                      fRule=CFM.IDM, fParam: dict = None, speed=0, interval=0.1) -> list:
        """
        该函数会清空对应车道上的车辆！！！

        通过输入车辆数和生成方式，快速添加车辆，从上游向下游生成。

        参数含义详情请见Lane.addDriver函数
        """
        # 使用前清空该车道上的车辆
        self.delDriver(self._driverList)
        # 默认车俩种类为passenger
        carLength = Obstacle.getStatic(carType, V_STATIC.LENGTH)
        maxCarNum = np.floor(self.length / carLength)
        if carNum > maxCarNum:
            TrasimError(vem.CANNOT_PLACE_CAR.format(f"此车道carNum最多为{maxCarNum}"))
        gap = self.length / carNum
        rand = random.Random(self.car_produce_seed)
        for i in range(carNum):
            if method == 'uniform':
                self.addDriver(gap * i, fRule=fRule, fParam=fParam, speed=speed, tool_mode=True, interval=interval)
            elif method == 'uniform_speed':
                r = rand.random()
                self.addDriver(gap * i, fRule=fRule, fParam=fParam, speed=r, tool_mode=True, interval=interval)
        self.updateDriverRelation()
        return [driver.ID for driver in self._driverList]

    def addMultiDrivers(self, driverDictList: Sequence[dict]):
        """
        向此车道放置多辆车，此函数优化了放置大量车辆的程序性能

        :_param driverDictList: 车辆参数列表，详情见Lane.addDriver函数
        :return: 车辆ID列表
        """
        if len(driverDictList) == 0:
            return []
        driver_id_list = []
        for driverDict in driverDictList:
            driverDict['multi_mode'] = True
            driver_id_list.append(self.addDriver(**driverDict))
        # self.updateDriverRelation()
        return driverDictList

    def addDriver(self, xOffset: int | float, carType: V_TYPE = V_TYPE.PASSENGER,
                  speed: int | float = 0, acc: int | float = 0, carLength: int | float = 5,
                  fRule=CFM.IDM, fParam: dict = None, **kwargs) -> str:
        """
        向此车道添加单个车辆

        :_param xOffset: 上游起点为0，下游终点为当前车道长度
        :_param carType: 车辆类型，支持constant.V_TYPE中的类型
        :_param speed: 车辆速度
        :_param acc: 车辆加速度
        :_param carLength: 车辆长度
        :_param fRule: 车辆跟驰规则，支持constant.CFM中的类型
        :_param fParam: 车辆跟驰参数
        :_param kwargs: 其他参数
        :return: 车辆ID
        """
        driver = Vehicle(self, carType=carType, mode=self._mode)
        driver.vehicle.dynamic[V_DYNAMIC.X_OFFSET] = xOffset
        driver.setDynamic(V_DYNAMIC.VELOCITY, speed)
        driver.setDynamic(V_DYNAMIC.ACC, acc)
        driver.setDynamic('roadID', self.ID)
        driver.vehicle.static[V_STATIC.LENGTH] = carLength
        driver.maxOffset = self.length
        driver.setfRule(fRule)
        driver.interval = kwargs.get('interval', 0.1)
        if fParam is not None:
            driver.setfParam(fParam)
        if kwargs.get('historyOn', False):
            maxHistory = kwargs.get('maxHistory', False)
            if maxHistory:
                driver.historyOn()
            else:
                driver.historyOn(maxHistory=maxHistory)
        # 车辆插入过程
        if self.first is None:
            self.first = driver
            self.last = driver
            driver.leader = driver
            driver.follower = driver
        else:
            if kwargs.get('tool_mode', False):
                self.first = driver
                driver.leader = self._driverList[0]
                driver.follower = self._driverList[-1]
                driver.leader.follower = driver
                driver.follower.leader = driver
            elif kwargs.get('multi_node', False):
                pass
            else:
                self._driverInsert(driver)
        driver.selfCheck()  # 检查车辆初始化属性有无问题
        self._driverList.append(driver)
        return driver.ID

    def _driverInsert(self, driver: Vehicle):
        """
        插入车辆，

        :_param driver:
        :return:
        """
        xOffset = driver.getDynamic(V_DYNAMIC.X_OFFSET)
        length = driver.getStatic(V_STATIC.LENGTH)
        temp = self.last
        while xOffset >= temp.getDynamic(V_DYNAMIC.X_OFFSET):
            temp = temp.leader
            if temp == self.first:
                break
        if xOffset < temp.getDynamic(V_DYNAMIC.X_OFFSET):
            front_pos = temp.getDynamic(V_DYNAMIC.X_OFFSET) - temp.getStatic(V_STATIC.LENGTH)
            if temp.follower.ID != temp.ID:
                after_pos = temp.follower.getDynamic(V_DYNAMIC.X_OFFSET)
            else:
                after_pos = -np.inf
        else:
            front_pos = self.length
            after_pos = temp.getDynamic(V_DYNAMIC.X_OFFSET)

        if not (xOffset <= front_pos and after_pos <= xOffset - length):
            raise TrasimError(vem.CANNOT_PLACE_CAR.format('车辆重叠！'))
        elif xOffset < temp.getDynamic(V_DYNAMIC.X_OFFSET):
            if self.last == temp:
                self.last = driver
            driver.leader = temp
            driver.follower = temp.follower
            driver.follower.leader = driver
            temp.follower = driver
        else:
            self.first = driver
            driver.leader = temp.leader
            driver.follower = temp
            driver.leader.follower = driver
            temp.leader = driver

    def step(self, interval: float = 0.1, updateMethod: str = 'synchronous'):
        """updateMethod=synchronous(同步) or asynchronous(异步)"""
        temp = self.first
        first_id = self.first.ID
        while temp.follower.ID != first_id:
            temp.update(interval, updateMethod)
            temp = temp.follower
        temp.update(interval, updateMethod)  # 最后一辆车更新
        if updateMethod == 'synchronous':
            temp = self.first
            while temp.follower != first_id:
                temp.update(interval, updateMethod)
                temp = temp.follower
            temp.update(interval, updateMethod)  # 最后一辆车更新
        # self.updateDriverRelation()
        self._SIMULATION_STEP += 1
        Vehicle.SIMULATION_STEP = self._SIMULATION_STEP

    def collectCircleInfo(self):
        """
        记录本次位置循环车辆的信息
        :return:
        """
        pass_num = self.count.get('pass_num', 0)
        space_mean_speed: list = self.count.get('space_mean_speed', [])
        if self.last.isCircle is True:
            pass_num += 1
            space_mean_speed.append(1 / self.last.getDynamic(V_DYNAMIC.VELOCITY))
        self.count['pass_num'] = pass_num
        self.count['space_mean_speed'] = space_mean_speed

    def getCircleInfo(self):
        temp = self.count.copy()
        if len(temp['space_mean_speed']) == 0:
            temp['space_mean_speed'] = 0
        self.count.clear()
        return temp

    def setDriverfParam(self, driverID: str, paramDict: dict):
        """通过车辆ID修改单个车辆的跟驰参数"""
        driver = self._findDriver(driverID)[0]
        driver.setfParam(paramDict)

    def getDriverfParam(self, driverID: str):
        """通过车辆ID获取单个车辆的跟驰参数"""
        driver = self._findDriver(driverID)[0]
        return driver.getfParam()

    def getDriverfStatus(self, driverID: str):
        """获取对应车辆的跟驰状态"""
        driver = self._findDriver(driverID)[0]
        return driver.getfStatus()

    def takeOver(self, driverID: str, acc: int | float = 0):
        """the effect will show in next step"""
        driver, _ = self._findDriver(driverID)
        driver.setDynamicInOperation(acc)

    def takeOverPlus(self, driverID: str, xOffset: float, speed: float, acc: float):
        """the method is used to calibrate the cf model"""
        driver, _ = self._findDriver(driverID)
        driver.setDynamicInOperation(acc, xOffset=xOffset, speed=speed)

    def setHistory(self, driverID: str, maxHistory: int):
        if maxHistory > 0:
            driver, _ = self._findDriver(driverID)
            driver.historyOn(maxHistory=maxHistory)

    def getHistory(self, driverID: str):
        driver, _ = self._findDriver(driverID)
        if isinstance(driver, Vehicle):
            return driver.getHistory()
        return {}

    def getSpeed(self, driverID: str) -> float:
        """提供了更方便的单个车辆速度获取方法"""
        driver, _ = self._findDriver(driverID)
        return driver.getDynamic(V_DYNAMIC.VELOCITY)

    def getxOffset(self, driverID: str) -> float:
        """提供了更方便的单个车辆路线起点偏移获取方法"""
        driver, _ = self._findDriver(driverID)
        return driver.getDynamic(V_DYNAMIC.X_OFFSET)

    def getGapOrHeadway(self, driverID: str, isGap=True) -> float:
        """获取当前车与前车的净间距或者车头间距，仅在环形车道、单路段使用，
        isGap为True时为净间距，否则为车头间距"""
        driver, _ = self._findDriver(driverID)
        leader = driver.leader
        if driver.leader is not None:
            leaderX = leader.getDynamic(V_DYNAMIC.X_OFFSET)
            leaderL = leader.getStatic(V_STATIC.LENGTH)
            currentX = driver.getDynamic(V_DYNAMIC.X_OFFSET)
            if leaderX < currentX:
                return leaderX + self.length - currentX - isGap * leaderL
            return leaderX - currentX - isGap * leaderL
        else:
            return np.inf

    def getRelativeV(self, driverID: str):
        """获取相对速度，当前车减去前车速度"""
        driver, _ = self._findDriver(driverID)
        leader = driver.leader
        return driver.getDynamic(V_DYNAMIC.VELOCITY) - leader.getDynamic(V_DYNAMIC.VELOCITY)

    def getOrdIDList(self) -> list:
        """获取此车道所有车辆ID，以起点偏移量从小到大排列"""
        self._sortDriverList()
        return [driver.ID for driver in self._driverList]

    def getDriverDynamic(self, driverID: str | list, param: str | list):
        """根据driverID和param获取指定车辆的指定动态数据"""
        if isinstance(driverID, str):
            driver, _ = self._findDriver(driverID)
            return driver.getDynamic(param) if isinstance(param, str) else [driver.getDynamic(p) for p in param]
        else:
            dynamicDict = {}
            if isinstance(param, str):
                for ID in driverID:
                    driver, _ = self._findDriver(ID)
                    dynamicDict[ID] = driver.getDynamic(param)
                return dynamicDict
            else:
                for ID in driverID:
                    driver, _ = self._findDriver(ID)
                    dynamicDict[ID] = [driver.getDynamic(p) for p in param]
                return dynamicDict

    def getDriverStatic(self, driverID: str | list, param: str | list):
        """根据driverID和param获取指定车辆的指定静态数据"""
        if isinstance(driverID, str):
            driver, _ = self._findDriver(driverID)
            return driver.getStatic(param) if isinstance(param, str) else [driver.getStatic(p) for p in param]
        else:
            staticDict = {}
            if isinstance(param, str):
                for ID in driverID:
                    driver, _ = self._findDriver(ID)
                    staticDict[ID] = driver.getStatic(param)
                return staticDict
            else:
                for ID in driverID:
                    driver, _ = self._findDriver(ID)
                    staticDict[ID] = [driver.getStatic(p) for p in param]
                return staticDict




