# -*- coding = uft-8 -*-
# @Time : 2023-03-31 16:02
# @Author : yzbyx
# @File : has_ui.py
# @Software : PyCharm
import numpy as np
import pygame as pg
from typing import TYPE_CHECKING, Optional, Union

from pygame.time import Clock

from trasim_simplified.core.constant import MARKING_TYPE
from trasim_simplified.core.ui.sim2d_ui import UI

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.lane_abstract import LaneAbstract
    from trasim_simplified.core.frame.micro.road import Road

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)


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
        self.road_surface: Optional[pg.Surface] = None
        self.agent_surface: Optional[pg.Surface] = None
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

        self.road_surface = pg.Surface((self.screen_width, self.screen_height))
        self.agent_surface = pg.Surface((self.screen_width, self.screen_height), pg.SRCALPHA)

        self.draw_road()
        self.ui_update()

    def draw_road(self):
        self.road_surface.fill(WHITE)
        for lane in self.lane_list:
            rect = pg.Rect(0, lane.y_left, lane.lane_length, lane.y_right)
            pg.draw.rect(self.road_surface, GRAY, rect)
            if lane.marking_type is None or len(lane.marking_type[1]) == 0:
                pg.draw.line(
                    self.road_surface, YELLOW,
                    (0, lane.y_left), (lane.lane_length, lane.y_left), 2
                )
                pg.draw.line(
                    self.road_surface, YELLOW,
                    (0, lane.y_right), (lane.lane_length, lane.y_right), 2
                )
            else:
                for i in range(1, len(lane.marking_type[0]) - 1):
                    marker_type: MARKING_TYPE = lane.marking_type[1][i - 1]
                    pg.draw.line(
                        self.road_surface, WHITE if marker_type[0] == MARKING_TYPE.DASHED else YELLOW,
                        (0, lane.y_left), (lane.lane_length, lane.y_left), 2
                    )
                    pg.draw.line(
                        self.road_surface, WHITE if marker_type[0] == MARKING_TYPE.DASHED else YELLOW,
                        (0, lane.y_right), (lane.lane_length, lane.y_right), 2
                    )

    def ui_update(self):
        self.agent_surface.fill((0, 0, 0, 0))

        font = pg.font.SysFont('Times', 20)
        text = font.render(
            "steps: " + str(self.frame.step_), True, (255, 255, 255), None
        )
        self.screen.blit(text, (0, 0))

        for index, lane in enumerate(self.lane_list):
            for i, car in enumerate(lane.car_list):
                polygon = car.get_bbox()
                pg.draw.polygon(self.agent_surface, car.color, polygon, 1)

        pg.display.update()
        if self.frame_rate > 0:
            self.clock.tick(self.frame_rate)
