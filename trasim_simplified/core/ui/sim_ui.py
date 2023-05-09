# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:02
# @Author : yzbyx
# @File : has_ui.py
# @Software : PyCharm
import pygame as pg
from typing import TYPE_CHECKING, Optional

from pygame.time import Clock

if TYPE_CHECKING:
    from trasim_simplified.core.frame.lane_abstract import LaneAbstract


class UI:
    def __init__(self, frame_abstract: 'LaneAbstract'):
        self.frame = frame_abstract
        self.frame_rate = -1
        self.screen_width = 1980 / 1.5
        self.screen_height = 100 / 1.5
        self.screen: Optional[pg.Surface] = None
        self.clock: Optional[Clock] = None

    def ui_init(self, caption="微观交通流仿真", frame_rate=-1):
        pg.init()
        self.frame_rate = frame_rate
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        pg.display.set_caption(caption)
        self.clock = pg.time.Clock()

        self.ui_update()

    def ui_update(self):
        self.screen.fill((0, 0, 0))
        font = pg.font.SysFont('Times', 20)
        text = font.render("steps: " + str(self.frame.step_), True, (255, 255, 255), None)
        self.screen.blit(text, (0, 0))

        for i, car in enumerate(self.frame.car_list):
            # TODO: 实现转折或缩放
            # pos_y = int(self.car_pos[0, i] / 1000)
            # pos_x = self.car_pos[0, i] - pos_y * 1000
            pg.draw.rect(self.screen, (255, 0, 0),
                         (car.x / self.frame.lane_length * self.screen_width, int(self.screen_height / 2),
                          int(car.length / self.frame.lane_length * self.screen_width), 10))

        pg.display.update()
        if self.frame_rate > 0:
            self.clock.tick(self.frame_rate)
