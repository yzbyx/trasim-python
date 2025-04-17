# -*- coding: utf-8 -*-
# @time : 2023/6/3 17:01
# @Author : yzbyx
# @File : ctm_ui.py
# Software: PyCharm
from typing import Optional, TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from matplotlib.colors import Colormap
from pygame.time import Clock

if TYPE_CHECKING:
    from trasim_simplified.core.frame.macro.ctm_lane import CTM_Lane
    from trasim_simplified.core.frame.macro.ctm_road import CTM_Road


class CTM_UI:
    def __init__(self, ctm_instance: Union['CTM_Lane', 'CTM_Road']):
        self.ctm_instance = ctm_instance
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
        self.cmap: Colormap = plt.get_cmap('coolwarm')

        if not hasattr(self.ctm_instance, "lane_list"):
            self.lane_list = [self.ctm_instance]
        else:
            self.lane_list = self.ctm_instance.lane_list

    def ui_init(self, caption="宏观交通流仿真", frame_rate=-1):
        pg.init()
        self.frame_rate = frame_rate
        self.screen_height = int((int(self.ctm_instance.lane_length / 1000) + self.base_line_factor) *
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
        text = font.render("steps: " + str(self.ctm_instance.step_), True, (255, 255, 255), None)
        self.screen.blit(text, (0, 0))

        row_total = int(self.lane_list[0].lane_length / self.width_base) + 1
        for row in range(row_total):
            pos_y = start_y + row * self.single_height
            pg.draw.line(self.screen, [255, 255, 255],
                         [0, int(pos_y * self.height_scale)],
                         [int(self.width_base * self.width_scale), int(pos_y * self.height_scale)])
        for index, lane in enumerate(self.lane_list):
            cell_occ = lane.cell_occ
            for i, pos in enumerate(lane.cell_start_pos):
                row = int(pos / self.width_base)
                offset = start_y + lane.index * lane_width + row * self.single_height
                pos_y = int(offset * self.height_scale)
                pos_x = int((pos - int(pos / self.width_base) * self.width_base) * self.width_scale)
                pg.draw.rect(self.screen, list(np.round(np.array(self.cmap(cell_occ[i])) * 255).tolist())[:3],
                             (pos_x, pos_y, int(lane.cell_length[i] * self.width_scale),
                              int(lane_width * self.height_scale)))

        pg.display.update()
        if self.frame_rate > 0:
            self.clock.tick(self.frame_rate)
