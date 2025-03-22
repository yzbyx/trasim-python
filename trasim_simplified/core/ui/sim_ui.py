# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:02
# @Author : yzbyx
# @File : has_ui.py
# @Software : PyCharm
import pygame as pg
from typing import TYPE_CHECKING, Optional, Union

from pygame.time import Clock

from trasim_simplified.core.ui.sim2d_ui import UI

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract
    from trasim_simplified.core.frame.micro.road import Road


class UI2D(UI):
    def __init__(self, frame_abstract: Union['LaneAbstract', 'Road']):
        super().__init__(frame_abstract)
        self.frame_rate = -1
        self.width_base = 1000
        self.width_scale = 1.5
        self.height_scale = 2
        self.single_height = 20
        self.base_line_factor = 2
        self.screen_width = None
        self.screen_height = None
        self.screen: Optional[pg.Surface] = None
        self.clock: Optional[Clock] = None

        if not hasattr(self.frame, "lane_list"):
            self.lane_list = [self.frame]
        else:
            self.lane_list = self.frame.lane_list

    def ui_init(self, caption="微观交通流仿真", frame_rate=-1):
        pg.init()
        self.frame_rate = frame_rate
        self.screen_height = int((int(self.frame.lane_length / 1000) + self.base_line_factor) *
                                 self.single_height * self.height_scale)
        self.screen_width = int(self.width_base * self.width_scale)
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        pg.display.set_caption(caption)
        self.clock = pg.time.Clock()

        self.ui_update()

    def ui_update(self):
        self.screen.fill((0, 0, 0))

        lane_width = 5

        start_y = self.base_line_factor * self.single_height

        font = pg.font.SysFont('Times', 20)
        text = font.render("steps: " + str(self.frame.step_), True, (255, 255, 255), None)
        self.screen.blit(text, (0, 0))

        row_total = int(self.lane_list[0].lane_length / self.width_base) + 1
        for row in range(row_total):
            pos_y = start_y + row * self.single_height
            pg.draw.line(self.screen, [255, 255, 255],
                         [0, int(pos_y * self.height_scale)],
                         [int(self.width_base * self.width_scale), int(pos_y * self.height_scale)])
        for index, lane in enumerate(self.lane_list):
            for i, car in enumerate(lane.car_list):
                row = int(car.x / self.width_base)
                offset = start_y + lane.index * lane_width + row * self.single_height
                pos_y = int((offset + lane_width / 2 - car.width / 2) * self.height_scale)
                pos_x = int((car.x - int(car.x / self.width_base) * self.width_base) * self.width_scale)
                pg.draw.rect(self.screen, car.color,
                             (pos_x, pos_y, int(car.length * self.width_scale), int(car.width * self.height_scale)))

        pg.display.update()
        if self.frame_rate > 0:
            self.clock.tick(self.frame_rate)
