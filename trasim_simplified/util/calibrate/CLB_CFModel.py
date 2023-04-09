# -*- coding = uft-8 -*-
# @Time : 2022-04-06 23:24
# @Author : yzbyx
# @File : CLB_CFModel.py
# @Software : PyCharm
# -*- coding = uft-8 -*-
# @Time : 2022-02-11 9:21
# @Author : yzbyx
# @File : ga opt trasim.py
# @Software : PyCharm
import pickle
from multiprocessing.pool import Pool

import winsound

import geatpy as ea
import numpy as np
import pandas as pd

from core.constant import CFM, RUNMODE
from core.drawer import Drawer, drawer
from core.lane import Lane
from util.calibrate.hysteresisJudge import hysteresisJudge_revise_calibrate

from kinematics import cfm as cfName


def openPickle(file: str):
    with open(file, 'rb') as f:
        return pickle.load(f)


def simulation(currentV, currentY, leaderV, leaderY, param, output='fitness', leaderL=5, fRule=CFM.IDM):
    """根据param中的tau值确定仿真步长"""
    # 默认IDM
    predictY = []
    predictV = []
    leaderVList = []
    leaderYList = []

    currentV_ = []
    currentY_ = []
    leaderV_ = []
    leaderY_ = []
    tau = param.get('tau', False)
    interval = 0.1
    if tau:
        interval = tau / 10
        param['tau'] = interval
    lane = Lane(length=100000, mode=RUNMODE.SILENT)
    leaderID = lane.addDriver(xOffset=leaderY[0], speed=leaderV[0], fRule=fRule, carLength=leaderL)
    currentID = lane.addDriver(xOffset=currentY[0], speed=currentV[0], fRule=fRule)
    lane.setDriverfParam(currentID, param)

    step = int(interval / 0.1)
    preStep = 0
    isBreak = False
    for i, _ in enumerate(leaderV):
        if preStep + step != i:
            continue
        preStep = i
        lane.takeOverPlus(leaderID, leaderY[i], leaderV[i], (leaderV[i] - leaderV[i - step]) / interval)
        lane.step(interval=interval)
        if currentV[i] < 1e-10:
            currentV_.append(currentV[i] + 1e-10)
        else:
            currentV_.append(currentV[i])
        currentY_.append(currentY[i])
        leaderV_.append(leaderV[i])
        leaderY_.append(leaderY[i])

        predictY.append(lane.getxOffset(currentID))  # 得出预测的Y值，共有len(self._leaderV)-1个
        predictV.append(lane.getSpeed(currentID))
        leaderVList.append(lane.getSpeed(leaderID))
        leaderYList.append(lane.getxOffset(leaderID))

        if np.isnan(predictY[-1]) or np.isnan(predictV[-1]) or np.isinf(predictV[-1]) or np.isinf(predictY[-1]):
            isBreak = True
            break

    fitness = -1e10
    predictY_np = np.array(predictY)
    predictV_np = np.array(predictV)

    realY_np = np.array(currentY_)
    realV_np = np.array(currentV_)

    leaderY_np = np.array(leaderY_)

    # ---车头时距也可作为合适的性能指标----
    if not isBreak:
        # 计算RMSPE指标，Root Mean Square Percentage Errors
        # deltaY = predictY_np - realY_np
        # errorY = np.sqrt((np.sum(np.power(deltaY, 2))) / np.sum(np.power(realY_np, 2)))

        headway_predict = predictY_np - leaderY_np
        headway_real = realY_np - leaderY_np
        delta_headway = headway_predict - headway_real
        errorHeadway = np.sqrt((np.sum(np.power(delta_headway, 2))) / np.sum(np.power(headway_real, 2)))

        deltaV = predictV_np - realV_np
        errorV = np.sqrt((np.sum(np.power(deltaV, 2))) / np.sum(np.power(realV_np, 2)))

        fitness = - (errorV + errorHeadway)
    if np.isnan(fitness) or np.isinf(fitness):
        fitness = -1e10
    if output == 'fitness':
        return fitness
    elif output == 'predict':
        return predictY_np, predictV_np, np.array(leaderYList), np.array(leaderVList), realY_np, realV_np
    else:
        return None


class CLB_CFModel:
    Range = {
        CFM.IDM: {'Dim': 6, 'varTypes': [0, 0, 1, 0, 0, 0],
                  'lb': [0, 0, 1, 0.1, 0.1, 0.1], 'ub': [40, 10, 10, 5, 5, 5]},
        CFM.GIPPS: {'Dim': 6, 'varTypes': [0, 0, 0, 1, 0, 0],
                    'lb': [0.1, -5, 0, 3, 5, -5], 'ub': [5, -0.1, 40, 30, 15, -0.1]},  # tau*10并取整
        # CFM.NON_LINEAR_GHR: {'Dim': 4, 'varTypes': [0, 0, 0, 1],
        #                      'lb': [-10, 0, 0, 3], 'ub': [10, 10, 60, 30]},  # tau*10并取整
        CFM.NON_LINEAR_GHR: {'Dim': 4, 'varTypes': [0, 0, 0, 1],
                             'lb': [-5, -5, -10, 3], 'ub': [5, 5, 10, 30]},  # tau*10并取整
        CFM.WIEDEMANN_99: {'Dim': 12, 'varTypes': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           'lb': [0, 0.1, 0.1, -20, -5, 0.1, 0.1, 0, 0.1, 0.1, 0, 0],
                           'ub': [20, 5, 10, -0.1, -0.1, 5, 20, 1, 8, 8, 40, 0.5]},
        CFM.OPTIMAL_VELOCITY: {'Dim': 5, 'varTypes': [0, 0, 0, 0, 0],
                               'lb': [0.2, 0, 0.05, 5, 0], 'ub': [10, 40, 0.2, 50, 5]}
    }
    Name = {
        CFM.IDM: ['v0', 's0', 'delta', 'T', 'omega', 'd'],
        CFM.GIPPS: list(cfName.GIPPS.DEFAULT_PARAM.keys()),
        CFM.NON_LINEAR_GHR: list(cfName.GHR.DEFAULT_PARAM.keys()),
        CFM.WIEDEMANN_99: list(cfName.W99.DEFAULT_PARAM.keys()),
        CFM.OPTIMAL_VELOCITY: list(cfName.OVM.DEFAULT_PARAM.keys())
    }

    def __init__(self, fModel=CFM.IDM):
        hysteresisDataByType = {'strong': [], 'weak': [], 'negligible': [], 'negative': []}
        for laneID in range(1, 9):
            path = f'E:/pythonscript/g-project/data/NGSIM/us-101-vehicle-trajectory-data/' \
                   f'0750am-0805am/data/hysteresisDataByType_{str(laneID)}.pkl'
            temp = openPickle(path)
            hysteresisDataByType['strong'].extend(temp['strong'])
            hysteresisDataByType['weak'].extend(temp['weak'])
            hysteresisDataByType['negligible'].extend(temp['negligible'])
            hysteresisDataByType['negative'].extend(temp['negative'])
        self.hysteresisDataByType = hysteresisDataByType

        self.fModel = fModel
        self.paramProblem = CLB_CFModel.Range[self.fModel]
        self.fParam = CLB_CFModel.Name[self.fModel]

        self.targetType = None
        self.index = 0
        self.seed = 0
        self.posRange = None
        self.leaderY = None
        self.leaderV = None
        self.leaderL = None
        self.currentY = None
        self.currentV = None
        self.vehicleID = None
        self.interval = 0.1
        self.saveRes = False

        self.res = {}

    def run(self, hType, index, seed, plot=False, interval=0.1, saveRes=False, onlyLoadData=False):
        self.targetType = hType
        self.index = index
        self.seed = seed
        self.interval = interval
        self.saveRes = saveRes
        # [currentY, currentV, leaderY, leaderV, posRange, decOrAccJudge, c, rangeIndex, v, strength, leaderL]
        self.leaderY = self.hysteresisDataByType[self.targetType][self.index][2]
        self.leaderV = self.hysteresisDataByType[self.targetType][self.index][3]
        self.leaderL = self.hysteresisDataByType[self.targetType][self.index][10]
        self.currentY = self.hysteresisDataByType[self.targetType][self.index][0]
        self.currentV = self.hysteresisDataByType[self.targetType][self.index][1]
        self.vehicleID = str(self.hysteresisDataByType[self.targetType][self.index][8])
        self.posRange = self.hysteresisDataByType[self.targetType][self.index][4]
        self.posRange = str(self.posRange[0]) + '-' + str(self.posRange[1]) + '-' + str(self.posRange[2])

        if not onlyLoadData:
            self.solveProblem()
            if plot:
                self.plot()
            return self.res

    def solveProblem(self):
        @ea.Problem.single
        def evalVars(Vars_):  # 定义目标函数（含约束）
            param = {}
            for i, item in enumerate(Vars_):
                param[self.fParam[i]] = item
            tau = param.get('tau', False)
            if tau:
                self.interval = tau / 10
            f = simulation(self.currentV, self.currentY, self.leaderV, self.leaderY, param, output='fitness',
                           leaderL=self.leaderL, fRule=self.fModel)
            return f

        problem = ea.Problem(name='test',
                             M=1,  # 目标维数
                             maxormins=[-1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                             Dim=self.paramProblem['Dim'],  # 决策变量维数
                             varTypes=self.paramProblem['varTypes'],  # 决策变量的类型列表，0：实数；1：整数
                             lb=self.paramProblem['lb'],  # 决策变量下界
                             ub=self.paramProblem['ub'],  # 决策变量上界
                             evalVars=evalVars)
        # 构建算法
        algorithm = ea.soea_SEGA_templet(problem,
                                         ea.Population(Encoding='RI', NIND=30),
                                         MAXGEN=300,  # 最大进化代数。
                                         logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                         trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                         maxTrappedCount=50)  # 进化停滞计数器最大上限值。
        algorithm.mutOper.Pm = 0.5  # 变异概率
        algorithm.recOper.XOVR = 0.7  # 重组概率
        # 求解
        self.res = ea.optimize(algorithm, verbose=False, seed=self.seed, drawing=0, outputMsg=False, drawLog=False,
                               saveFlag=False)
        print(f"{self.fModel} {self.seed}: \t{self.res['ObjV'][0]},"
              f" \t{self.res['Vars'][0]}, \t{self.res['executeTime']}")
        if self.saveRes:
            np.save('./data/res_seed-' + str(self.seed) + '_' + self.vehicleID + '_' + self.posRange[0], self.res)

    def plot(self, Vars=None, hType=None, index=None):
        if Vars is None:
            Vars = self.res['Vars'][0]
        else:
            Vars = np.array(Vars)
            self.run(hType, index, -1, onlyLoadData=True)

        param = {}
        for i, item in enumerate(Vars):
            param[self.fParam[i]] = item

        predictY, predictV, leaderY, leaderV, currentY, currentV = simulation(self.currentV, self.currentY,
                                                                              self.leaderV, self.leaderY,
                                                                              param,
                                                                              output='predict', leaderL=self.leaderL,
                                                                              fRule=self.fModel)

        deltaV = [0]
        preV = predictV[0]
        for index in range(len(predictV) - 1):
            deltaV.append(predictV[index + 1] - preV)
            preV = predictV[index + 1]

        drawer = Drawer()
        figIDList = drawer.initFigure(figNum=4)
        drawer.setAxParam(figIDList[0], xLabel='Frame(0.1s)', yLabel='Speed(m/s)', title='最佳子代速度')
        drawer.myPlot(figIDList[0], dataX=1, dataY=predictV, pattern='b-')
        drawer.myPlot(figIDList[0], dataX=1, dataY=currentV, pattern='r-')

        drawer.setAxParam(figIDList[1], xLabel='PredictV', yLabel='predictSpace', title='Loop')
        drawer.myScatter(figIDList[1], dataX=predictV, dataY=leaderY - predictY, color=deltaV, label='V-Space-predict')
        drawer.setAxParam(figIDList[3], xLabel='CurrentV', yLabel='Space', title='Loop')
        drawer.myScatter(figIDList[3], dataX=currentV, dataY=leaderY - currentY,
                         color=np.insert(np.array(currentV[1:]) - np.array(currentV[:-1]), 0, 0),
                         label='V-Space-real')

        drawer.setAxParam(figIDList[2], xLabel='Time(0.1s)', yLabel='Location(m)', title='time-space map')
        drawer.myPlot(figIDList[2], dataX=1, dataY=leaderY, pattern='r-', label='leader')
        drawer.myPlot(figIDList[2], dataX=1, dataY=currentY, pattern='g-', label='current')
        drawer.myPlot(figIDList[2], dataX=1, dataY=predictY, pattern='b-', label='current-predict')

        drawer.saveFigure(figIDList[0],
                          './figure/predictV' + str(self.seed) + '_' + self.vehicleID + '_' + self.posRange[0] + '.svg',
                          delPre=True)
        drawer.saveFigure(figIDList[1],
                          './figure/predictLoop' + str(self.seed) + '_' + self.vehicleID + '_' + self.posRange[0] +
                          '.svg',
                          delPre=True)
        drawer.saveFigure(figIDList[2],
                          './figure/predictY' + str(self.seed) + '_' + self.vehicleID + '_' + self.posRange[0] + '.svg',
                          delPre=True)
        drawer.saveFigure(figIDList[3],
                          './figure/realLoop' + str(self.seed) + '_' + self.vehicleID + '_' + self.posRange[0] + '.svg',
                          delPre=True)
        # np.save('./data/param_seed-' + str(self.seed) + '_' + self.vehicleID + '_' + self.posRange, _param)


def _singleRun(clb, key, index, plot=False):
    temp = []
    objV = - np.Inf
    for seed in range(0, 5):
        r = clb.run(hType=key, index=index, seed=seed, plot=plot)
        value = np.array(r['ObjV'][0][0])
        if value > objV:
            temp = r['Vars'][0]
            objV = value
    return temp, objV


def CLB_Main(cfm, plot=False, indexFix=None):
    clb = CLB_CFModel(cfm)
    name = clb.Name[cfm]
    df_data = {'Vehicle_ID': [], 'Start_Loc': [], 'ObjV': []}
    for n in name:
        df_data[n] = []

    num = 0
    for j, key in enumerate(clb.hysteresisDataByType.keys()):
        for index in range(len(clb.hysteresisDataByType[key])):
            if isinstance(indexFix, int):
                if num == indexFix:
                    temp, objV = _singleRun(clb, key, index, plot=plot)
                    df: pd.DataFrame = pd.read_pickle('./data/df_' + cfm)
                    for i, n in enumerate(name):
                        if n == 'tau':
                            df.at[indexFix, name] = temp[i] / 10
                        else:
                            df.at[indexFix, name] = temp[i]
                    df.at[indexFix, 'ObjV'] = objV
                    df.to_pickle('./data/df_' + cfm)
                    return
                else:
                    num += 1
                    continue
            num += 1
            if num < 88 and cfm == CFM.WIEDEMANN_99:
                continue
            if num == 88 and cfm == CFM.WIEDEMANN_99:
                df: pd.DataFrame = pd.read_pickle('./data/df_' + cfm)
                for n in list(df.columns):
                    df_data[n] = df[n].to_list()
                continue

            d = clb.hysteresisDataByType[key][index]
            _, _, _, _, posRange, decOrAccJudge, c, rangeIndex, v, strength, leaderL = d
            temp, objV = _singleRun(clb, key, index, plot)
            for i, n in enumerate(name):
                if n == 'tau':
                    df_data[n].append(temp[i] / 10)
                else:
                    df_data[n].append(temp[i])
            df_data['Start_Loc'].append(posRange[0])
            df_data['Vehicle_ID'].append(v)
            df_data['ObjV'].append(objV)
            print(f'{cfm} final: \t{temp}, \t{objV}')

            df = pd.DataFrame(data=df_data)
            df.to_pickle('./data/df_' + cfm)


def CAL_Hysteresis(cfm):
    clb = CLB_CFModel(cfm)
    name = clb.Name[cfm]
    df: pd.DataFrame = pd.read_pickle('./data/df_' + cfm)
    HType = ['strong', 'weak', 'negligible', 'negative']
    dataAll = []
    for t in HType:
        dataAll.extend(clb.hysteresisDataByType[t])

    for index in df.index:
        dataDF = df.iloc[index].to_dict()
        fParam = {}
        for n in name:
            if n == 'tau':
                fParam[n] = round(dataDF[n] * 10)
            else:
                fParam[n] = dataDF[n]
        for i, data in enumerate(dataAll):
            currentY, currentV, leaderY, leaderV, posRange, decOrAccJudge, c, rangeIndex, v, strength, leaderL = data
            if v == dataDF['Vehicle_ID'] and posRange[0] == dataDF['Start_Loc']:
                predictY, predictV, leaderY_, leaderV_, realY, realV = simulation(currentV, currentY, leaderV, leaderY,
                                                                                  fParam.copy(),
                                                                                  output='predict', leaderL=leaderL,
                                                                                  fRule=cfm)
                if v == 2354:
                    draw(realY, leaderY_, predictY, realV, leaderV_, predictV, fParam.get('tau', 1) / 10, v)
                gap = leaderY_ - predictY - leaderL
                print(len(gap), len(rangeIndex[0]), len(rangeIndex[1]))
                posChange = posRange[1] - posRange[0]
                tau = fParam.get('tau', 1)  # 这个地方的tau是乘以10后的
                # step = tau
                # posChange = round(posChange / step)
                # if decOrAccJudge == -1:
                #     gapAcc = gap[posChange:]
                #     gapDec = gap[:posChange]
                #     vAcc = predictV[posChange:]
                #     vDec = predictV[:posChange]
                # else:
                #     gapAcc = gap[:posChange]
                #     gapDec = gap[posChange:]
                #     vAcc = predictV[:posChange]
                #     vDec = predictV[posChange:]
                print(v, posRange[0])
                dec2acc = True if decOrAccJudge == -1 else False
                c_, rangeIndex_, strength_ = hysteresisJudge_revise_calibrate([gap, predictV], dec2acc,
                                                                              leaderL=leaderL)
                print(c_, strength_)
                # gapAvg, QDelta, minV, maxV
                if c_ > -2:
                    df.at[index, 'Dec_Or_Acc'] = c_
                    df.at[index, 'Q_Delta'] = strength_[1]
                    df.at[index, 'Gap_Avg'] = strength_[0]
                    df.at[index, 'Min_V'] = strength_[2]
                    df.at[index, 'Max_V'] = strength_[3]
                else:
                    df.at[index, 'Dec_Or_Acc'] = c_
                    df.at[index, 'Q_Delta'] = np.NAN
                    df.at[index, 'Gap_Avg'] = np.NAN
                    df.at[index, 'Min_V'] = np.NAN
                    df.at[index, 'Max_V'] = np.NAN
                df.at[index, 'Time_Acc'] = len(rangeIndex_[0]) * tau * 0.1
                df.at[index, 'Time_Dec'] = len(rangeIndex_[1]) * tau * 0.1
                df.at[index, 'Time'] = len(rangeIndex_[0]) * tau * 0.1 + len(rangeIndex_[1]) * tau * 0.1
                break
    df.to_pickle('./data/df_' + cfm + '_clb')


def draw(currentY, leaderY, predictY, realV, leaderV_, predictV, interval, name='default'):
    drawer.setAxParam(xLabel='速度(m/s)', yLabel='车头间距(m)', fontSize=14)
    drawer.myPlot(dataX=realV, dataY=leaderY - currentY,
                  pattern='b--', label='实际后车轨迹')
    drawer.myPlot(dataX=predictV, dataY=leaderY - predictY,
                  pattern='r-', label='标定后车轨迹 拟合优度: -0.038')
    drawer.saveFigure(path=f'GHR_轨迹对比_{name}.svg')


def addLocSize(cfm):
    df: pd.DataFrame = pd.read_pickle('./data/df_' + cfm + '_clb')
    clb = CLB_CFModel(cfm)
    HType = ['strong', 'weak', 'negligible', 'negative']
    dataAll = []
    for t in HType:
        dataAll.extend(clb.hysteresisDataByType[t])
    for index in df.index:
        for i, data in enumerate(dataAll):
            currentY, currentV, leaderY, leaderV, posRange, decOrAccJudge, c, rangeIndex, v, strength, leaderL = data
            if v == df.at[index, 'Vehicle_ID'] and posRange[0] == df.at[index, 'Start_Loc']:
                df.at[index, 'Loc_Size'] = posRange[-1] - posRange[0] + 1
                break
    df.to_pickle('./data/df_' + cfm + '_clb_sized')


def multiProcess(cfmList):
    pool = Pool(processes=2)
    pool.map(CLB_Main, cfmList)
    pool.close()
    pool.join()
    pool.map(CAL_Hysteresis, cfmList)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # clb = CLB_CFModel(CFM.NON_LINEAR_GHR)
    # clb.plot(Vars=[-0.479813, 0.317554, 2.826557, 2.1], hType='strong', index=1)
    # l = [CFM.IDM, CFM.NON_LINEAR_GHR]
    # CLB_Main(CFM.NON_LINEAR_GHR)
    CAL_Hysteresis(CFM.NON_LINEAR_GHR)
    # addLocSize(CFM.IDM)
    # multiProcess(l)
    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
