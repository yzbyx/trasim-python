# -*- coding = uft-8 -*-
# @Time : 2023-03-25 22:37
# @Author : yzbyx
# @File : frame.py
# @Software : PyCharm
import abc
from abc import ABC
from typing import Optional, TYPE_CHECKING

import numpy as np

from trasim_simplified.core.constant import SECTION_TYPE, V_TYPE, CFM
from trasim_simplified.core.data.data_container import DataContainer
from trasim_simplified.core.data.data_processor import DataProcessor
from trasim_simplified.core.ui.sim_ui import UI
from trasim_simplified.core.vehicle import Vehicle
from trasim_simplified.core.data.data_container import Info as C_Info
from trasim_simplified.msg.trasimError import TrasimError

if TYPE_CHECKING:
    from trasim_simplified.core.frame.road import Road


class LaneAbstract(ABC):
    def __init__(self, lane_length: float):
        self.ID = 0
        self.index = 0
        self.road: Optional[Road] = None
        self.default_speed_limit = 30.
        self.car_num_total = 0
        self.is_circle = None
        self.lane_length = float(lane_length)
        self.section_type: dict[str, list[float, float]] = {}
        self.speed_limit: dict[float, list[float, float]] = {}

        self.id_accumulate = 0
        self.car_num_list: list[int] = []
        self.car_type_list: list[str] = []
        self.car_length_list: list[float] = []
        self.car_initial_speed_list: list[float] = []
        self.speed_with_random_list: list[bool] = []
        self.cf_name_list: list[str] = []
        self.cf_param_list: list[dict] = []
        self.car_param_list: list[dict] = []
        self.lc_name_list: list[str] = []
        self.lc_param_list: list[dict] = []
        self.lc_out_list: list[Vehicle] = []
        self.lc_add_list: list[Vehicle] = []

        self.car_list: list[Vehicle] = []
        self._dummy_car_list: list[Vehicle] = []
        self.out_car_has_data: list[Vehicle] = []

        self.step_ = 0
        """当前仿真步次"""
        self.time_ = 0
        """当前仿真时长 [s]"""
        self.yield_ = True
        """run()是否为迭代器"""
        self.road_control = False
        """是否为Road类控制"""

        self.dt = 0.1
        """仿真步长 [s]"""
        self.warm_up_step = int(5 * 60 / self.dt)
        """预热时间 [s]"""
        self.sim_step = int(10 * 60 / self.dt)
        """总仿真步 [次]"""

        self.data_save = False
        self.data_container: DataContainer = DataContainer(self)
        self.data_processor: DataProcessor = DataProcessor(self)

        self.has_ui = False
        self.ui: UI = UI(self)

    def _get_new_car_id(self):
        if not self.road_control:
            self.id_accumulate += 1
            return self.id_accumulate
        else:
            return self.road.get_new_car_id()

    def set_section_type(self, type_: str, start_pos: float = -1, end_pos: float = -1):
        if start_pos < 0:
            start_pos = 0
        if end_pos < 0:
            end_pos = self.lane_length
        self.section_type.update({type_: [start_pos, end_pos]})

    def get_section_type(self, pos) -> set[str]:
        type_ = set()
        if len(self.section_type) == 0:
            type_.add(SECTION_TYPE.BASE)
        for key in self.section_type.keys():
            pos_ = self.section_type[key]
            if pos_[0] <= pos < pos_[1]:
                type_.add(key)
        return type_

    def set_speed_limit(self, speed_limit=30, start_pos=-1, end_pos=-1):
        assert speed_limit >= 0
        if start_pos < 0:
            start_pos = 0
        if end_pos < 0:
            end_pos = self.lane_length
        self.speed_limit.update({speed_limit: [start_pos, end_pos]})

    def get_speed_limit(self, pos):
        if len(self.speed_limit) == 0:
            return self.default_speed_limit
        for key in self.speed_limit.keys():
            pos_ = self.speed_limit[key]
            if pos_[0] <= pos <= pos_[1]:
                return key
        return self.default_speed_limit

    @property
    def car_num(self):
        return len(self.car_list)

    def car_config(self, car_num: int, car_length: float, car_type: str, car_initial_speed: int,
                   speed_with_random: bool, cf_name: str, cf_param: dict[str, float], car_param: dict,
                   lc_name: Optional[str] = None, lc_param: Optional[dict[str, float]] = None):
        """如果是开边界，则car_num与car_loader配合可以代表车型比例，如果car_loader中的flow为复数，则car_num为真实生成车辆数"""
        self.car_num_list.append(car_num)
        self.car_length_list.append(car_length)
        self.car_type_list.append(car_type)
        self.car_initial_speed_list.append(car_initial_speed)
        self.speed_with_random_list.append(speed_with_random)
        self.cf_name_list.append(cf_name)
        self.cf_param_list.append(cf_param)
        self.car_param_list.append(car_param)
        self.lc_name_list.append(lc_name)
        self.lc_param_list.append(lc_param)

    def car_load(self, car_gap=-1):
        car_num_total = sum(self.car_num_list)
        car_length_total = np.sum(np.array(self.car_num_list) * np.array(self.car_length_list))
        gap = (self.lane_length - car_length_total) / car_num_total
        assert gap >= 0, f"该密度下，车辆重叠！"

        x = 0
        car_count = 0
        car_type_index_list = []
        for i in range(len(self.car_num_list)):
            car_type_index_list.extend([i] * self.car_num_list[i])
        np.random.shuffle(car_type_index_list)

        for index, i in enumerate(car_type_index_list):
            vehicle = Vehicle(self, self.car_type_list[i], self._get_new_car_id(), self.car_length_list[i])
            vehicle.x = x
            vehicle.v = np.random.uniform(
                max(self.car_initial_speed_list[i] - 0.5, 0), self.car_initial_speed_list[i] + 0.5
            ) if self.speed_with_random_list[i] else self.car_initial_speed_list[i]
            vehicle.a = 0
            vehicle.set_cf_model(self.cf_name_list[i], self.cf_param_list[i])
            vehicle.set_lc_model(self.lc_name_list[i], self.lc_param_list[i])
            vehicle.set_car_param(self.car_param_list[i])

            self.car_list.append(vehicle)
            if index != car_num_total - 1:
                length = self.car_length_list[car_type_index_list[index + 1]]
                if car_gap < 0:
                    x = x + gap + length
                else:
                    x = x + car_gap + length
            car_count += 1

        if len(self.car_list) > 2:
            for i, car in enumerate(self.car_list[1: -1]):
                car.leader = self.car_list[i + 2]
                car.follower = self.car_list[i]
            self.car_list[0].leader = self.car_list[1]
            self.car_list[-1].follower = self.car_list[-2]
        else:
            self.car_list[0].leader = self.car_list[-1]
            self.car_list[-1].follower = self.car_list[0]

        if self.is_circle is True:
            self.car_list[0].follower = self.car_list[-1]
            self.car_list[-1].leader = self.car_list[0]

    def run(self, data_save=True, has_ui=True, **kwargs):
        if kwargs is None:
            kwargs = {}
        self.data_save = data_save
        """是否记录数据"""
        self.warm_up_step = kwargs.get("warm_up_step", int(5 * 60 / self.dt))
        """预热步数 [s]"""
        self.dt = kwargs.get("dt", 0.1)
        """仿真步长 [s]"""
        self.sim_step = kwargs.get("sim_step", int(10 * 60 / self.dt))
        """总仿真步 [次]"""
        frame_rate = kwargs.get("frame_rate", -1)
        """pygame刷新率 [fps]"""
        caption = kwargs.get("ui_caption", "微观交通流仿真")
        self.yield_ = kwargs.get("if_yield", True)
        """run()是否为迭代器"""
        self.has_ui = has_ui

        if self.has_ui and not self.road_control:
            self.ui.ui_init(caption=caption, frame_rate=frame_rate)

        # 整个仿真能够运行sim_step的仿真步
        while self.sim_step != self.step_:
            if not self.is_circle:
                self.car_summon()
            # 能够记录warm_up_step仿真步时的车辆数据
            if self.data_save and self.step_ >= self.warm_up_step:
                self.record()
            self.step()  # 未更新状态，但已经计算跟驰结果
            # 控制车辆对应的step需要在下一个仿真步才能显现到数据记录中
            if self.yield_: yield self.step_
            self.update_state()  # 更新车辆状态
            if self.road_control: yield self.step_
            self.step_ += 1
            self.time_ += self.dt
            if self.has_ui and not self.road_control: self.ui.ui_update()

    def car_state_update_common(self, car: Vehicle):
        car_speed_before = car.v
        car.v += car.cf_acc * self.dt

        if car.v > car.cf_model.get_expect_speed():
            expect_speed = car.cf_model.get_expect_speed()
            car.a = (expect_speed - car.v) / self.dt
            car.v = expect_speed
        elif car.v < 0:
            # if car.v < - 1e-3:
            #     print("存在速度为负的车辆！")
            car.cf_acc = - (car_speed_before / self.dt)
            car.v = 0
        else:
            car.a = car.cf_acc

        car.x += (car_speed_before + car.v) * self.dt / 2

        if car.leader is not None and car.leader.type == V_TYPE.OBSTACLE:
            if car.x > car.leader.x:
                car.x = car.leader.x

    @abc.abstractmethod
    def update_state(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    def car_summon(self):
        """用于开边界车辆生成"""
        pass

    def record(self):
        for car in self.car_list:
            if car.type != V_TYPE.OBSTACLE:
                car.record()

    def _make_dummy_car(self, pos):
        car = Vehicle(self, V_TYPE.OBSTACLE, -1, 0.1)
        car.set_cf_model(CFM.DUMMY, {})
        car.x = pos + 0.1
        car.v = 0
        car.a = 0
        return car

    def set_block(self, pos):
        dummy_car = self._make_dummy_car(pos)
        self.car_insert_by_instance(dummy_car)

    def get_appropriate_car(self) -> int:
        """获取合适进行控制扰动的单个车辆，（未到车道一半线的最近车辆）"""
        pos = self.lane_length / 2
        car_pos = np.array([car.x for car in self.car_list])
        pos_ = np.where(car_pos < pos)[0]
        max_pos = np.argmax(car_pos[pos_])
        return self.car_list[pos_[max_pos]].ID

    def take_over(self, car_id: int, acc_values: float):
        """控制指定车辆运动"""
        for car in self.car_list:
            if car.ID == car_id:
                car.cf_acc = acc_values

    def car_insert(self, car_length: float, car_type: str, car_pos: float, car_speed: float, car_acc: float,
                   cf_name: str, cf_param: dict[str, float], car_param: dict,
                   lc_name: Optional[str] = None, lc_param: Optional[dict[str, float]] = None):
        car = self._make_car(car_length, car_type, car_pos, car_speed, car_acc,
                             cf_name, cf_param, car_param, lc_name, lc_param)
        self.car_insert_by_instance(car)
        return car.ID

    def car_remove(self, car: Vehicle, put_out_car_has_data=False):
        if put_out_car_has_data:
            self.out_car_has_data.append(car)
        self.car_list.remove(car)
        if car.leader is not None:
            car.leader.follower = car.follower
        if car.follower is not None:
            car.follower.leader = car.leader

    def _make_car(self, car_length, car_type, car_pos, car_speed, car_acc,
                  cf_name, cf_param, car_param, lc_name, lc_param):
        car = Vehicle(self, car_type, self._get_new_car_id(), car_length)
        car.set_cf_model(cf_name, cf_param)
        car.set_lc_model(lc_name, lc_param)
        car.set_car_param(car_param)
        car.x = car_pos
        car.v = car_speed
        car.a = car_acc
        return car

    def car_insert_by_instance(self, car: Vehicle, is_dummy=False):
        car.lane = self
        car.leader = car.follower = None
        if len(self.car_list) != 0:
            pos_list = np.array([car.x for car in self.car_list])
            index = np.where(pos_list < car.x)[0]
            if len(index) != 0:
                index = index[-1]
                follower = self.car_list[index]
            else:
                index = -1
                follower = self.car_list[-1] if self.is_circle else None

            if follower is not None:
                leader = follower.leader

                follower.leader = car
                car.follower = follower
                car.leader = leader
                if leader is not None:
                    leader.follower = car
            else:
                car.leader = self.car_list[0]
                self.car_list[0].follower = car

            if not is_dummy:
                self.car_list.insert(index + 1, car)
        else:
            if not is_dummy:
                self.car_list.append(car)
            if self.is_circle:
                car.leader = car
                car.follower = car

    def get_car_info(self, id_: int, info_name: str):
        for car in self.car_list:
            if car.ID == id_:
                if info_name == C_Info.x:
                    return car.x
                if info_name == C_Info.v:
                    return car.v
                if info_name == C_Info.a:
                    return car.a
                if info_name == C_Info.gap:
                    return car.gap
                if info_name == C_Info.dhw:
                    return car.dhw
                raise TrasimError(f"{info_name}未创建！")

    def _get_car(self, id_):
        for car in self.car_list:
            if car.ID == id_:
                return car

    def car_insert_middle(self, car_length: float, car_type: str, car_speed: float, car_acc: float,
                          cf_name: str, cf_param: dict[str, float], car_param: dict, front_car_id,
                          lc_name: Optional[str] = None, lc_param: Optional[dict[str, float]] = None):
        follower_id = self.get_relative_id(front_car_id, -1)
        follower_pos = self.get_car_info(follower_id, C_Info.x)
        follower_gap = self.get_car_info(follower_id, C_Info.gap)
        follower_dhw = self.get_car_info(follower_id, C_Info.dhw)
        front_length = self._get_car(front_car_id).length

        if follower_dhw / 2 < car_length or follower_dhw / 2 < front_length:
            print("空间不足，插入失败！")
            return
        pos = follower_pos + follower_dhw / 2
        car = self._make_car(car_length, car_type, pos, car_speed, car_acc,
                             cf_name, cf_param, car_param, lc_name, lc_param)
        self.car_insert_by_instance(car)
        return car.ID

    def get_relative_id(self, id_, offset: int):
        """
        :param id_: 车辆ID
        :param offset: 正整数代表向下游检索，负数代表上游
        """
        return self._get_relative_car_by_id(id_, offset).ID

    def _get_relative_car_by_id(self, id_, offset: int):
        assert offset - int(offset) == 0, "offset必须是整数"
        for car in self.car_list:
            if car.ID == id_:
                while offset != 0:
                    if offset > 0:
                        if car.leader is not None:
                            car = car.leader
                        offset -= 1
                    else:
                        if car.follower is not None:
                            car = car.follower
                        offset += 1
                return car

    def get_relative_car(self, pos: float) -> tuple[Optional[Vehicle], Optional[Vehicle]]:
        """获取指定位置的前后车"""
        for car in self.car_list:
            if car.x > pos:
                return car.follower, car
        if len(self.car_list) != 0:
            return self.car_list[-1], self.car_list[0]
        return None, None

    def __str__(self):
        return "lane_length: " + str(self.lane_length) + \
            "\tcar_num: " + str(self.car_num_list) + \
            "\tcar_length: " + str(self.car_length_list) + \
            "\tcar_initial_speed: " + str(self.car_initial_speed_list) + \
            "\tbasic_record: " + str(self.data_save) + \
            "\thas_ui: " + str(self.has_ui) + \
            "\tframe_rate" + str(self.ui.frame_rate) + \
            "\tdt: " + str(self.dt) + \
            "\twarm_up_step: " + str(self.warm_up_step) + \
            "\tsim_step: " + str(self.sim_step)
