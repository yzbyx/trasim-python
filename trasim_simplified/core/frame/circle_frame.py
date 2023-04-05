# -*- coding = uft-8 -*-
# @Time : 2023-03-24 16:21
# @Author : yzbyx
# @File : circle_frame.py
# @Software : PyCharm
import numpy as np

from trasim_simplified.core.constant import CFM
from trasim_simplified.core.frame.frame_abstract import FrameAbstract
from trasim_simplified.util.decorator.mydecorator import timer_no_log


class FrameCircle(FrameAbstract):
    def __init__(self, lane_length: int, car_num: int, car_length: int, car_initial_speed: int, speed_with_random: bool,
                 cf_mode: str, cf_param: dict[str, float]):
        super().__init__(lane_length, car_num, car_length, car_initial_speed, speed_with_random, cf_mode, cf_param)

    def car_init(self):
        dhw = self.lane_length / self.car_num
        assert dhw >= self.car_length, f"该密度下，车辆重叠！此车身长度下车辆数最多为{np.floor(self.lane_length / self.car_length)}"
        self.car_pos = np.arange(0, self.lane_length, dhw).reshape(1, -1)
        assert self.car_num == self.car_pos.shape[1], f"车辆生成数量有误！目标：{self.car_num}，结果：{self.car_pos.shape}"
        if self.speed_with_random:
            self.car_speed = np.random.uniform(
                max(self.car_initial_speed - 0.5, 0),  self.car_initial_speed + 0.5, self.car_pos.shape
            ).reshape(1, -1)
        else:
            self.car_speed = (np.ones(self.car_pos.shape) * self.car_initial_speed).reshape(1, -1)
        self.car_acc = np.zeros(self.car_pos.shape).reshape(1, -1)

    def run(self, basic_save=True, has_ui=True, **kwargs):
        return super().run(basic_save, has_ui, **kwargs)

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


@timer_no_log
def run():
    _cf_param = {}
    take_over_index = 0
    warm_up_step = 6000
    sim = FrameCircle(1000, 20, 5, 1, False, CFM.IDM, _cf_param)
    for step in sim.run(basic_save=True, has_ui=False, warm_up_step=6000, sim_step=12000, dt=0.1):
        pass
        if warm_up_step + 100 == step:
            take_over_index = sim.get_appropriate_car()
        if warm_up_step + 100 < step <= warm_up_step + 120:
            sim.take_over(take_over_index, -3)
    sim.plot.basic_plot(((take_over_index - 1) if take_over_index != 0 else sim.car_num - 1))
    sim.data_processor.aggregate()
    sim.data_processor.aggregate_as_detect_loop(0, 995, 6000)
    sim.data_processor.print_result()

if __name__ == '__main__':
    run()
