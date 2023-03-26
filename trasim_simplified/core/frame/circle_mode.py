# -*- coding = uft-8 -*-
# @Time : 2023-03-24 16:21
# @Author : yzbyx
# @File : circle_mode.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.core.constant import CFM
from trasim_simplified.core.frame.frame_abstract import FrameAbstract
from trasim_simplified.util.decorator.mydecorator import timer_no_log


class FrameCircle(FrameAbstract):
    def __init__(self, lane_length: int, car_num: int, car_length: int, car_initial_speed: int, cf_mode: str,
                 cf_param: dict[str, float]):
        super().__init__(lane_length, car_num, car_length, car_initial_speed, cf_mode, cf_param)

    def car_init(self):
        dhw = self.lane_length / self.car_num
        assert dhw >= self.car_length, f"该密度下，车辆重叠！此车身长度下车辆数最多为{np.floor(self.lane_length / self.car_length)}"
        self.car_pos = np.arange(0, self.lane_length, dhw).reshape(1, -1)
        assert self.car_num == self.car_pos.shape[1], f"车辆生成数量有误！目标：{self.car_num}，结果：{self.car_pos.shape}"
        self.car_speed = np.random.uniform(
            max(self.car_initial_speed - 0.5, 0),  self.car_initial_speed + 0.5, self.car_pos.shape
        ).reshape(1, -1)
        self.car_acc = np.zeros(self.car_pos.shape).reshape(1, -1)

    @timer_no_log
    def run(self, basic_save=True, aggregate_cal=True, plot_data=True, df_save=True, ui=True, **kwargs):
        super().run(basic_save, aggregate_cal, plot_data, df_save, ui, **kwargs)

    def step(self):
        leader_x = np.roll(self.car_pos, -1)
        diff_x = leader_x - self.car_pos
        pos_ = np.where(diff_x < 0)
        leader_x[pos_] += self.lane_length
        self.car_acc = self.cf_model.step(
            self.car_speed,
            self.car_pos,
            np.roll(self.car_speed, -1),
            leader_x,
            self.car_length
        )
        car_speed_before = self.car_speed
        self.car_speed += self.car_acc * self.dt
        self.car_pos += (car_speed_before + self.car_speed) / 2 * self.dt
        self.car_pos[np.where(self.car_pos > self.lane_length)] -= self.lane_length

def run():
    _cf_param = {}
    sim = FrameCircle(1000, 60, 5, 0, CFM.IDM, _cf_param)
    sim.run(basic_save=True, ui=False, df_save=True, warm_up_step=3000, sim_step=3600, dt=1)


if __name__ == '__main__':
    run()
