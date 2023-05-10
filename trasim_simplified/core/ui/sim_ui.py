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
        self.width_base = 1000
        self.scale = 1.5
        self.single_height = 20
        self.screen_width = None
        self.screen_height = None
        self.screen: Optional[pg.Surface] = None
        self.clock: Optional[Clock] = None

    def ui_init(self, caption="微观交通流仿真", frame_rate=-1):
        pg.init()
        self.frame_rate = frame_rate
        self.screen_height = int((int(self.frame.lane_length / 1000) + 2) * self.single_height * self.scale)
        self.screen_width = int(self.width_base * self.scale)
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
            pos_y = int((int(car.x / 1000) + 2) * self.single_height) * self.scale
            pos_x = int((car.x - int(car.x / 1000) * 1000) * self.scale)
            pg.draw.rect(self.screen, car.color,
                         (pos_x, pos_y, int(car.length * self.scale), int(car.width * 2 * self.scale)))

        pg.display.update()
        if self.frame_rate > 0:
            self.clock.tick(self.frame_rate)
