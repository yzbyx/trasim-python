# -*- coding = uft-8 -*-
# @Time : 2022/1/12 18:29
# @Author : yzbyx
# @File : drawer.py
# @Software : PyCharm
import os
import time
import warnings
from copy import copy

from win32comext.shell import shell, shellcon

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mc
import matplotlib.cm as cm
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib import pyplot as plt, ticker


def get_current_time():
    ct = time.time()
    data_head = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    time_stamp = "%s.%s" % (data_head, str(ct).split('.')[-1][:3])
    return time_stamp


class Drawer:
    def __init__(self, figNum: int = None):
        self._fontSize = 10
        self._figIDList: list[str] = []
        self._haveColorBar: list[bool] = []
        self._dpi = 300

        self._currentUpper = None
        self._currentLower = None

        matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        plt.rcParams['savefig.facecolor'] = 'white'

        if figNum is not None:
            self.initFigure(figNum)

    @staticmethod
    def _dataXRemake(dataX, dataY):
        # 如果dataX没有设置，则默认以range(len(dataY))作为最终的dataX
        if dataX is None:
            dataX = np.arange(len(dataY))
        # 如果dataX设置为整数，则默认以range(len(dataY)) + dataX作为最终的dataX
        elif isinstance(dataX, int | float):
            dataX = np.array(range(len(dataY))) + dataX
        # 如果dataX为列表类型，则正常返回
        elif isinstance(dataX, list | np.ndarray | pd.Series):
            pass
        else:
            raise ValueError('The format of dataX is wrong!')
        return dataX

    def _normalization(self, color, Range):
        if Range is None:
            self._currentLower = min(color)
            self._currentUpper = max(color)
        elif isinstance(Range, tuple):
            self._currentLower = Range[0]
            self._currentUpper = Range[1]
        elif isinstance(Range, int | float):
            lower = abs(Range - min(color))
            upper = abs(max(color) - Range)
            r = max(lower, upper)
            self._currentLower = Range - r
            self._currentUpper = Range + r
        r = self._currentUpper - self._currentLower
        if r == 0:
            warnings.warn('color范围长度为0！', RuntimeWarning)
            return color
        return (color - self._currentLower) / r

    def _figSwitch(self, figID: str | int = None, axIndex: int = None) -> (plt.Figure, plt.Axes):
        if isinstance(figID, int):
            fig: plt.Figure = plt.figure(num=self._figIDList[figID])
        elif figID is None:
            if len(self._figIDList) == 0:
                self._figIDList.extend(self.initFigure())
            fig: plt.Figure = plt.gcf()
        else:
            fig: plt.Figure = plt.figure(num=figID)
        if axIndex is None:
            ax: plt.Axes = fig.gca()
        else:
            ax: plt.Axes = fig.axes[axIndex]
        return fig, ax

    @staticmethod
    def _getCurrentFigID():
        manager = plt.get_current_fig_manager()
        return manager.get_window_title()

    def _drawColorBar(self, cmap=None):
        """此方法需要在切换画布(ax)之前调用"""
        fig, ax = self._figSwitch(None, None)
        lower = self._currentLower
        upper = self._currentUpper

        # 生成colorbar
        norm = mc.Normalize(vmin=lower, vmax=upper)
        cmap = cmap if cmap is not None else cm.get_cmap('rainbow')
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.02, use_gridspec=False)
        # 设置colorbar标签
        tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
        cb.locator = tick_locator
        ticks = np.arange(lower, upper + upper / 8, (upper - lower) / 4)
        decNum = len(str(upper).split('.')[-1]) if len(str(upper).split('.')) != 1 else 0
        ticks.round(decimals=decNum)
        cb.set_ticks(list(ticks), fontdict={'size': self._fontSize})
        cb.update_ticks()

        manager = plt.get_current_fig_manager()
        figID = manager.get_window_title()
        self._haveColorBar[self._figIDList.index(figID)] = True

    def saveFigure(self, figID: str | int = None, path: str = None, dpi: int = None, delPre=False, date=False):
        """保存图像，需要写文件类型后缀，自动添加日期，最后关闭图像，delPre为删除之前path相同且由此函数保存的图片"""
        # 将plt切换到指定的fig并返回画布对象
        fig, _ = self._figSwitch(figID)
        if path is None:
            path = os.path.join(os.path.expanduser('~'), "Desktop", "untitled.png")
        f = path.split('.')[-1]
        prefix = path[:-len('.' + f)]
        if date:
            prefix += '%'
            prefix += get_current_time()
            # 删除之前的图片
            if delPre:
                absPath = os.path.abspath(path)
                pathList = absPath.split('\\')
                head = pathList[0]
                fileName = pathList.pop(-1)
                pathList.pop(0)
                for item in pathList:
                    head += '\\' + item
                for file in os.listdir(head):
                    haveChar = False
                    if file[0] == '%':
                        haveChar = True
                    trueName = '.'.join(['%' * haveChar + '%'.join(file.split('%')[:-1]), f])
                    if trueName == fileName and len(file) == len(fileName) + len('%' + get_current_time()):
                        fileAbsPath = os.path.join(head, file)
                        shell.SHFileOperation((0, shellcon.FO_DELETE, fileAbsPath, None,
                                               shellcon.FOF_SILENT | shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,
                                               None, None))
                        print(f'{file} 已放入回收站')
        # 保存图片
        if dpi is not None:
            self._dpi = dpi
        plt.savefig(prefix + '.' + f, dpi=self._dpi, bbox_inches='tight')
        plt.close(fig)

    def initFigure(self, figNum: int = 1, axesShape=(1, 1), shareXY=(False, False),
                   figSize: (int, int) = (7, 5), figureIDList: list[str] = None):
        """初始化图幅"""
        # 以title作为是否多图的基准
        IDList = []
        for i in range(figNum):
            # 初始化画板
            if isinstance(figureIDList, list):
                suffix = str(figureIDList[i])
            else:
                suffix = str(len(self._figIDList))
            # 设置画板的ID
            figID = 'fig_' + suffix
            if figID in self._figIDList:
                raise ValueError(f'图像名"{suffix}"已有')
            # 在画板上添加画布
            fig, _ = plt.subplots(axesShape[0], axesShape[1], sharex=shareXY[0], sharey=shareXY[1],
                                  num=figID, figsize=figSize)
            self._haveColorBar.append(False)
            self._figIDList.append(figID)
            IDList.append(figID)
        # 将当前plt指向第一个fig
        plt.figure(self._figIDList[0])
        return IDList

    def setAxParam(self, figID: str | int = None, axIndex: int = None,
                   xLabel: str = None, yLabel: str = None, title: str = None,
                   tickX: list | tuple = None, tickY: list | tuple = None,
                   xLimit: list = None, yLimit: list = None,
                   scientific: (bool, bool) = (False, False), fontSize: int = 12,
                   gridOn=False):
        # 将plt切换到指定的fig并返回画布对象
        fig, ax = self._figSwitch(figID, axIndex)

        self._fontSize = fontSize
        # 设置刻度字体大小
        plt.xticks(fontsize=self._fontSize)
        plt.yticks(fontsize=self._fontSize)
        # 设置标题
        plt.title(title, fontdict={'fontsize': self._fontSize + 2})
        # 设置画布的坐标轴标签
        ax.set_xlabel(xLabel, fontsize=self._fontSize)
        ax.set_ylabel(yLabel, fontsize=self._fontSize)
        # 设置网格
        if gridOn:
            ax.grid(visible=True)
        # 设置画布坐标轴是否科学计数
        ax.xaxis.major.formatter.set_scientific(scientific[0])
        ax.yaxis.major.formatter.set_scientific(scientific[1])

        if xLimit is not None:
            ax.set_xlim(left=xLimit[0], right=xLimit[1])
        if yLimit is not None:
            ax.set_ylim(bottom=yLimit[0], top=yLimit[1])

        if xLabel is None or yLabel is None:
            fig.align_labels()

        if tickX is not None:
            ax.xaxis.set_ticks(tickX[0], tickX[1])
        if tickY is not None:
            ax.yaxis.set_ticks(tickY[0], tickY[1])

    def myPlot(self, figID: str | int | plt.Figure = None, axIndex: int | plt.Axes = None,  # 目标画板和画布
               dataX=None, dataY=None,  # 基本数据
               pattern='r-', label=None, isLegend=True, lineWidth=1, labelPos: str = 'best', pattern_no_color='-',  # 基础样式
               color=None, steps=1, cmap: cm = None, barOn=False, barRange: tuple | float | int = None):  # 绘制渐变线使用
        """
        绘制折线图, dataX可为None、实数、列表, color为相邻数据点之间的线段颜色, 如果多次plot且提供color，需要填写barRange

        upper left
        """
        # 将plt切换到指定的fig并返回对象
        if isinstance(figID, plt.Figure) and isinstance(axIndex, plt.Axes):
            fig, ax = figID, axIndex
        else:
            fig, ax = self._figSwitch(figID, axIndex)
            ax: plt.Axes = ax

        dataX = self._dataXRemake(dataX, dataY)
        lc = None

        # if len(dataY) < 2:
        #     raise ValueError(f'dataY长度{len(dataY)}小于2，无法绘制！\n'
        #                      f'dataY: {dataY}')

        if color is not None:
            # 绘制渐变色线
            if len(color) == len(dataX):
                message = f"color: {len(color)}, dataX: {len(dataX)}, 不符合中间连线的数量，默认删除color的第一个值"
                warnings.warn(message, RuntimeWarning)
                # 多一位，删除最前面的一个数据
                color = np.asarray(color)
                color = np.delete(color, 0)
            elif len(color) != len(dataX) - 1:
                raise ValueError(f'color长度{len(color)}未对应数据长度{len(dataX)}')
            color = np.asarray(color)

            path = mpath.Path(np.column_stack([dataX, dataY]))
            v = path.interpolated(steps=steps).vertices
            prePoint = v[:-1]
            nowPoint = v[1:]
            seg = np.array([(a, b) for a, b in zip(prePoint, nowPoint)])

            # color的切分暂时借用matplotlib中的interpolated方法
            path = mpath.Path(np.column_stack([np.arange(len(color)), color]))
            color: np.ndarray = path.interpolated(steps=steps).vertices[:, 1]
            colorNormal = self._normalization(color, barRange)
            if cmap is None:
                cmap = cm.get_cmap('rainbow')
            newMap = mc.LinearSegmentedColormap.from_list(
                'newMap', cmap(np.linspace(min(colorNormal), max(colorNormal), 100)))  # 100

            lc = mcoll.LineCollection(seg, array=colorNormal, cmap=newMap,
                                      linewidths=lineWidth, linestyles=pattern_no_color)
            lc.set_label(label)

            ax.add_collection(lc)
            ax.autoscale(True)
            # ax.legend(loc=label, fontsize=40)

            if figID is None:
                figID = self._getCurrentFigID()
                figID = self._figIDList.index(figID)
            elif isinstance(figID, str):
                figID = self._figIDList.index(figID)
            if cmap is not None and barOn and self._haveColorBar[figID] is False:
                self._drawColorBar(cmap=cmap)
        else:
            # 普通绘制
            lc = ax.plot(dataX, dataY, pattern, label=label, linewidth=lineWidth)
        # 设置标签自适应、字体大小、标签边框显示
        if label is not None and isLegend:
            ax.legend(loc=labelPos, fontsize=self._fontSize - 1, frameon=False)

        return lc

    def myScatter(self, figID: str | int = None, axIndex: int = None,  # 目标画板和画布
                  dataX=None, dataY=None,  # 基本数据
                  pattern='o', size=1, label=None,  # 基础样式
                  color=None, cmap: cm = None, barOn=False, barRange: tuple | float | int = None):  # 绘制彩色散点和colorbar
        """绘制散点图，dataX可为None、实数、列表, 如果多次scatter且提供color，需要填写barRange"""
        # 将plt切换到指定的fig并返回画布对象
        fig, ax = self._figSwitch(figID, axIndex)
        ax: plt.Axes = ax

        dataX = self._dataXRemake(dataX, dataY)

        if isinstance(color, list | tuple | np.ndarray):
            if not isinstance(color[0], list | tuple):
                colorNormal, newMap = self._colorNormalize(color, cmap, barRange)

                ax.scatter(dataX, dataY, s=size, c=colorNormal, cmap=newMap, marker=pattern, label=label)
            else:
                ax.scatter(dataX, dataY, s=size, c=color, marker=pattern, label=label)
        else:
            ax.scatter(dataX, dataY, s=size, c=color, marker=pattern, label=label)

        if figID is None:
            figID = self._getCurrentFigID()
            figID = self._figIDList.index(figID)
        elif isinstance(figID, str):
            figID = self._figIDList.index(figID)
        if barOn and self._haveColorBar[figID] is False:
            self._drawColorBar(cmap=cmap)
        if label is not None:
            ax.legend(loc='best', fontsize=self._fontSize - 2, frameon=False)

    def myImShow(self, figID: str | int = None, axIndex: int = None,  # 目标画板和画布
                 mat=None, cmap: cm = None, aspect: int | float = 1, **kwargs):
        """通过矩阵绘制图像, **kwargs参数详见ax.imshow"""
        fig, ax = self._figSwitch(figID, axIndex)
        ax: plt.Axes = ax
        if cmap is None:
            cmap = cm.get_cmap('rainbow')
        ax.imshow(mat, cmap=cmap, aspect=aspect, **kwargs)

    def myBoxPlot(self, figID: str | int = None, axIndex: int = None,  # 目标画板和画布
                  x=None, labels=None, showFliers=True):
        fig, ax = self._figSwitch(figID, axIndex)
        if labels is None and isinstance(x, pd.DataFrame):
            labels = x.columns
        ax.boxplot(x, labels=labels, showfliers=showFliers)

    def myPColorMesh(self, figID: str | int = None, axIndex: int = None,
                     dataX=None, dataY=None, dataValue=None,
                     cmap: cm = None, barOn=False, barRange: tuple | float | int = None, autoNormalize=True):
        """绘制2D热力图"""
        fig, ax = self._figSwitch(figID, axIndex)

        dataValue = np.array(dataValue, dtype=np.float64)
        if len(dataValue.shape) != 2:
            raise ValueError('dataValue维数必须为二维！')
        if not autoNormalize:
            nrows, ncols = dataValue.shape
            dataValue = dataValue.reshape((nrows * ncols,))
            colorNormal, newMap = self._colorNormalize(dataValue, cmap, barRange)
            colorNormal = colorNormal.reshape((nrows, ncols))

            ax.pcolormesh(dataX, dataY, colorNormal, cmap=newMap)

            if figID is None:
                figID = self._getCurrentFigID()
                figID = self._figIDList.index(figID)
            elif isinstance(figID, str):
                figID = self._figIDList.index(figID)
            if barOn and self._haveColorBar[figID] is False:
                self._drawColorBar(cmap=cmap)
        else:
            c = ax.pcolormesh(dataX, dataY, dataValue, cmap=cmap)
            fig.colorbar(c)

    def myContourf(self, figID: str | int = None, axIndex: int = None,
                   dataX=None, dataY=None, dataValue=None, stageNum=8,
                   cmap: cm = None, barOn=False, barRange: tuple | float | int = None):
        """绘制等值线图，注意dataX、dataY需要与值的索引对应，即dataX[index1] dataY[index2] -> dataValue[index1, index2]"""
        fig, ax = self._figSwitch(figID, axIndex)

        C = ax.contour(dataX, dataY, dataValue, stageNum, colors='black')
        c = ax.contourf(dataX, dataY, dataValue, stageNum * 10, cmap=plt.cm.get_cmap('coolwarm').reversed())
        ax.clabel(C, inline=1, fontsize=10)

        fig.colorbar(c)

    def myQuiver(self, figID: str | int = None, axIndex: int = None,
                 dataX=None, dataY=None, arrowX=None, arrowY=None,
                 color='b', pivot="tail", scale=1.0, scale_units='xy', width=1,
                 alpha=1, angles='xy', units='inches', autoNormalize=False,
                 cmap: cm = None, barOn=False, barRange: tuple | float | int = None):
        """绘制场图"""
        fig, ax = self._figSwitch(figID, axIndex)

        dataValue = np.array(color, dtype=np.float64)
        if len(dataValue.shape) != 2:
            raise ValueError('dataValue维数必须为二维！')
        if not autoNormalize:
            nrows, ncols = dataValue.shape
            dataValue = dataValue.reshape((nrows * ncols,))
            colorNormal, newMap = self._colorNormalize(dataValue, cmap, barRange)
            colorNormal = colorNormal.reshape((nrows, ncols))

            ax.quiver(dataX, dataY, arrowX, arrowY, colorNormal, pivot=pivot, scale=scale,
                      units=units, angles=angles, alpha=alpha, width=width, cmap=newMap, scale_units=scale_units)

            if figID is None:
                figID = self._getCurrentFigID()
                figID = self._figIDList.index(figID)
            elif isinstance(figID, str):
                figID = self._figIDList.index(figID)
            if barOn and self._haveColorBar[figID] is False:
                self._drawColorBar(cmap=cmap)
        else:
            ax.quiver(dataX, dataY, arrowX, arrowY, color, pivot=pivot, scale=scale,
                      units=units, angles=angles, alpha=alpha, width=width, cmap=cmap, scale_units=scale_units)

    @staticmethod
    def myPause(t: int = 1000):
        plt.pause(t)

    def _colorNormalize(self, color, cmap=None, barRange=None):
        colorNormal = copy(color)
        newMap = None
        if isinstance(color, str) is False and color is not None:
            color = np.asarray(color)
            colorNormal = self._normalization(color, barRange)
            if cmap is None:
                cmap = cm.get_cmap('rainbow')
            newMap = mc.LinearSegmentedColormap.from_list(
                'newMap', cmap(np.linspace(min(colorNormal), max(colorNormal), 100)))  # 100
        elif color is None:
            colorNormal = None
            newMap = None
        return colorNormal, newMap

    def getFigInstance(self, figID: str | int = None) -> (plt.Figure, plt.Axes):
        """需要更多自定义功能，直接返回fig对象"""
        return self._figSwitch(figID)


drawer = Drawer()


def _test_myPlot_myScatter():
    x1 = np.linspace(0, 4 * np.pi, 100)
    y1 = np.sin(x1) * 2
    y2 = np.cos(x1)
    drawer = Drawer(figNum=2)

    deltaY1 = np.insert(y1[1:] - y1[:-1], 0, 0, axis=0)
    deltaY1[0] = deltaY1[-1]
    deltaY2 = np.insert(y2[1:] - y2[:-1], 0, 0, axis=0)
    deltaY2[0] = deltaY2[-1]

    bar_Range = (min(min(deltaY2), min(deltaY1)), max(max(deltaY2), max(deltaY1)))

    drawer.setAxParam(0, xLabel='x', yLabel='y')
    drawer.myPlot(dataX=x1, dataY=y1, color=deltaY1, cmap=cm.get_cmap('coolwarm'), barRange=bar_Range)
    drawer.myPlot(dataX=x1, dataY=y2, color=deltaY2, cmap=cm.get_cmap('coolwarm'), barRange=bar_Range)

    drawer.setAxParam(1, xLabel='x', yLabel='y')
    drawer.myScatter(dataX=x1, dataY=y1, color=deltaY1, barRange=bar_Range)
    drawer.myScatter(dataX=x1, dataY=y2, color=deltaY2, barRange=bar_Range)

    plt.pause(1000)


if __name__ == '__main__':
    _test_myPlot_myScatter()
