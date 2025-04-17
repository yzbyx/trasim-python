# -*- coding: utf-8 -*-
# @time : 2023/9/30 17:27
# @Author : yzbyx
# @File : pyqtgraph_ui.py
# Software: PyCharm
from typing import Union, Optional

try:
    import pyqtgraph as pg
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QApplication
    from pyqtgraph import GraphicsLayoutWidget
except ImportError:
    pass

from trasim_simplified.core.agent.vehicle import Vehicle


class PyqtUI:
    def __init__(self, frame_abstract: Union['LaneAbstract', 'Road']):
        self.frame = frame_abstract
        self.frame_rate = -1
        self.width_base = 1000
        self.width_scale = 1.5
        self.height_scale = 2
        self.single_height = 20
        self.base_line_factor = 2
        self.screen_width = None
        self.screen_height = None
        self.screen: Optional[GraphicsLayoutWidget] = None
        self.clock = None

        if not hasattr(self.frame, "lane_list"):
            self.lane_list = [self.frame]
        else:
            self.lane_list = self.frame.lane_list

        self.lane_width = 5
        self.start_y = self.base_line_factor * self.single_height
        self.text: Optional[pg.TextItem] = pg.TextItem("", color=(255, 255, 255))

    def ui_init(self, caption="微观交通流仿真", frame_rate=-1):
        self.frame_rate = frame_rate
        self.screen_height = int((int(self.frame.lane_length / 1000) + self.base_line_factor) *
                                 self.single_height * self.height_scale)
        self.screen_width = int(self.width_base * self.width_scale)
        win: GraphicsLayoutWidget = pg.GraphicsLayoutWidget(show=True)
        win.resize(self.screen_width, self.screen_height)
        win.setWindowTitle(caption)

        self.screen = win
        self.screen.setBackground('w')
        self.screen.addItem(self.text)
        self.text.setText("steps: " + str(self.frame.step_))
        self.draw_line()

        # self.screen.render()
        #
        # timer = QtCore.QTimer()
        # timer.timeout.connect(self.ui_update)
        # timer.start(int(1 / frame_rate * 1000))

    def draw_line(self):
        row_total = int(self.lane_list[0].lane_length / self.width_base) + 1
        for row in range(row_total):
            pos_y = self.start_y + row * self.single_height
            self.screen.addItem(pg.InfiniteLine(int(pos_y * self.height_scale), angle=0, pen=(255, 255, 255)))

    def ui_update(self):
        self.text.setText("steps: " + str(self.frame.step_))

        for index, lane in enumerate(self.lane_list):
            for i, car in enumerate(lane.car_list):
                car: Vehicle = car
                row = int(car.x / self.width_base)
                offset = self.start_y + lane.index * self.lane_width + row * self.single_height
                pos_y = int((offset + self.lane_width / 2 - car.width / 2) * self.height_scale)
                pos_x = int((car.x - int(car.x / self.width_base) * self.width_base) * self.width_scale)

                if not hasattr(car, "plot_item"):
                    carItem = pg.ButtonItem(width=car.width)
                    carItem.setPos(pos_x, pos_y)
                    carItem.setBrush(pg.mkBrush(car.color))
                    car.__setattr__("plot_item", carItem)
                    car.__setattr__("screen", self.screen)
                else:
                    carItem = car.__getattribute__("plot_item")
                    carItem.setPos(pos_x, pos_y)
                    self.screen.addItem(carItem)
        self.screen.render()
