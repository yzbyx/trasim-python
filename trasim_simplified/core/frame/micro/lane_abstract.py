# -*- coding = uft-8 -*-
# @time : 2023-03-25 22:37
# @Author : yzbyx
# @File : frame.py
# @Software : PyCharm
import abc
from abc import ABC
from typing import Optional, TYPE_CHECKING, Union

import numpy as np

from trasim_simplified.core.agent import get_veh_class
from trasim_simplified.core.constant import SECTION_TYPE, V_TYPE, CFM, MARKING_TYPE, V_CLASS, RouteType
from trasim_simplified.core.data.data_container import DataContainer
from trasim_simplified.core.data.data_processor import DataProcessor
from trasim_simplified.core.ui.sim_ui import UI2D
from trasim_simplified.core.agent.vehicle import Vehicle
from trasim_simplified.core.data.data_container import Info as C_Info
from trasim_simplified.msg.trasimError import TrasimError

if TYPE_CHECKING:
    from trasim_simplified.core.frame.micro.road import Road


class LaneAbstract(ABC):
    def __init__(self, lane_length: float, speed_limit: float = 30, width: float = 3.5):
        self.ID = 0
        self.index = 0
        self.add_num = 0
        self.road: Optional[Road] = None
        self.left_neighbour_lane: Optional[LaneAbstract] = None
        self.right_neighbour_lane: Optional[LaneAbstract] = None

        self._default_speed_limit = speed_limit
        self.car_num_total = 0
        self.is_circle = None
        self.lane_length = float(lane_length)
        self.width = width
        self.heading = 0
        self.section_types: Optional[list[list[float], list[SECTION_TYPE]]] = None
        """车道分段类型，n+1个纵向坐标点，n个分段类型"""
        self.marking_type: Optional[list[list[float], list[tuple[MARKING_TYPE, MARKING_TYPE]]]] = None
        """车道标线类型，n+1个纵向坐标点，n个标线类型"""
        self.speed_limit: Optional[list[list[float], list[float]]] = None
        """车道限速，n+1个纵向坐标点，n个限速值"""

        self.id_accumulate = 0
        self.car_num_list: list[int] = []
        self.car_type_list: list[V_TYPE] = []
        self.car_class_list: list[V_CLASS] = []
        self.car_length_list: list[float] = []
        self.car_initial_speed_list: list[float] = []
        self.speed_with_random_list: list[bool] = []
        self.cf_name_list: list[str] = []
        self.cf_param_list: list[dict] = []
        self.car_param_list: list[dict] = []
        self.lc_name_list: list[str] = []
        self.lc_param_list: list[dict] = []
        self.destination_lanes_list: list = []
        self.route_type_list: list[RouteType] = []

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
        self.force_speed_limit = False
        """是否强制车辆速度不超过道路限速"""
        self.state_update_method = "Euler"  # TODO
        """状态更新方式：
        
        1.
        Euler欧拉 (v(t + Δt) += a(t) * Δt, x(t + Δt) += v(t + Δt) * Δt),
        
        2.
        Ballistic抛物线 (v(t + Δt) += a(t) * Δt, x(t + Δt) += (v(t) + v(t + Δt)) * Δt / 2), 
        
        3. 
        Treiber, M., Kanagaraj, V., 2015.
         Comparing numerical integration schemes for time-continuous car-following models. Physica A 419, 183–195.
        
        Trapezoidal梯形 (v(t + Δt) += 0.5(a(t) + a(t + Δt)) * Δt, x(t + Δt) += (v(t) + v(t + Δt)) * Δt / 2)
        
        4. the standard fourth-order Runge–Kutta method (RK4)
        ...
        """

        """仿真步长 [s]"""
        self.warm_up_step = int(5 * 60 / self.dt)
        """预热时间 [s]"""
        self.sim_step = int(10 * 60 / self.dt)
        """总仿真步 [次]"""

        self.data_save = False
        self.data_container: DataContainer = DataContainer(self)
        self.data_processor: DataProcessor = DataProcessor()

        self.has_ui = False
        self.ui: UI2D = UI2D(self)

        self.y_center = 0
        self.y_left = self.width / 2
        self.y_right = - self.width / 2

    @property
    def dt(self):
        return 0.1

    def update_y_info(self):
        """更新车道信息"""
        self.y_center = - self.add_num * self.width - self.width / 2
        self.y_left = self.y_center + self.width / 2
        self.y_right = self.y_center - self.width / 2

    def _get_new_car_id(self):
        if not self.road_control:
            self.id_accumulate += 1
            return self.id_accumulate
        else:
            return self.road.get_new_car_id()

    def set_section_type(self, section_type: list[SECTION_TYPE], x_lon: list[float] = None):
        """设置车道分段类型"""
        if x_lon is None:
            x_lon = [0, self.lane_length]
        assert len(x_lon) - 1 == len(section_type)
        self.section_types = [x_lon, section_type]

    def get_section_type(self, x_lon) -> SECTION_TYPE:
        if self.section_types is None:
            return SECTION_TYPE.BASE

        for i in range(len(self.section_types[0]) - 1):
            if self.section_types[0][i] <= x_lon < self.section_types[0][i + 1]:
                return self.section_types[1][i]

        return SECTION_TYPE.BASE

    def set_marking_type(self, marking_type: list[tuple[MARKING_TYPE, MARKING_TYPE]], x_lon: list[float]):
        """设置车道标线类型"""
        if x_lon is None:
            x_lon = [0, self.lane_length]
        assert len(x_lon) - 1 == len(marking_type)
        self.marking_type = [x_lon, marking_type]

    def get_marking_type(self, pos) -> tuple[MARKING_TYPE, MARKING_TYPE]:
        """获取车道标线类型（左标线、右标线）"""
        if self.marking_type is None:
            return MARKING_TYPE.SOLID, MARKING_TYPE.SOLID

        for i in range(len(self.marking_type[0]) - 1):
            if self.marking_type[0][i] <= pos < self.marking_type[0][i + 1]:
                return self.marking_type[1][i]

        return MARKING_TYPE.SOLID, MARKING_TYPE.SOLID

    def set_speed_limit(self, speed_limit, x_lon=None):
        """设置车道限速"""
        if x_lon is None:
            x_lon = [0, self.lane_length]
        if isinstance(speed_limit, (int, float)):
            speed_limit = [speed_limit]
        assert len(x_lon) - 1 == len(speed_limit)
        self.speed_limit = [x_lon, speed_limit]

    def get_speed_limit(self, pos) -> float:
        """获取车道限速"""
        if self.speed_limit is None:
            return self._default_speed_limit

        for i in range(len(self.speed_limit[0]) - 1):
            if self.speed_limit[0][i] <= pos < self.speed_limit[0][i + 1]:
                return self.speed_limit[1][i]

        return self._default_speed_limit

    @property
    def car_num(self):
        return len(self.car_list)

    def car_config(self, car_num: Union[int, float], car_length: float, car_type: V_TYPE, car_class: V_CLASS,
                   car_initial_speed: float, speed_with_random: bool,
                   cf_name: str, cf_param: dict[str, float], car_param: dict,
                   lc_name: Optional[str] = None, lc_param: Optional[dict[str, float]] = None,
                   destination_lanes=tuple[int], route_type=None):
        """如果是开边界，则car_num与car_loader配合可以代表车型比例，如果car_loader中的flow为复数，则car_num为真实生成车辆数"""
        if 0 < car_num < 1:
            car_num = int(np.floor(self.lane_length * car_num / car_length))
        self.car_num_list.append(car_num)
        self.car_length_list.append(car_length)
        self.car_type_list.append(car_type)
        self.car_class_list.append(car_class)
        self.car_initial_speed_list.append(car_initial_speed)
        self.speed_with_random_list.append(speed_with_random)
        self.cf_name_list.append(cf_name)
        self.cf_param_list.append(cf_param)
        self.car_param_list.append(car_param)
        self.lc_name_list.append(lc_name)
        self.lc_param_list.append(lc_param)
        self.destination_lanes_list.append(destination_lanes)
        self.route_type_list.append(route_type)

    def car_load(self, car_gap=-1, jam_num=-1):
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

        num_node = []  # 记录分段阻塞对应的头车index
        jam_gap = None
        if jam_num > 1:
            car_num = int(np.floor(car_num_total / jam_num))
            jam_gap = (self.lane_length - car_length_total) / jam_num
            for j in range(jam_num - 1):
                num_node.append(car_num * (j + 1))

        for index, i in enumerate(car_type_index_list):
            VehClass = get_veh_class(self.car_class_list[i])
            vehicle = VehClass(self, self.car_type_list[i], self._get_new_car_id(), self.car_length_list[i])
            vehicle.set_cf_model(self.cf_name_list[i], self.cf_param_list[i])
            vehicle.set_lc_model(self.lc_name_list[i], self.lc_param_list[i])
            if self.car_initial_speed_list[i] < 0:
                self.car_initial_speed_list[i] = vehicle.cf_model.get_expect_speed()
            vehicle.x = x
            vehicle.v = np.random.uniform(
                max(self.car_initial_speed_list[i] - 0.5, 0), self.car_initial_speed_list[i] + 0.5
            ) if self.speed_with_random_list[i] else self.car_initial_speed_list[i]
            vehicle.a = 0
            vehicle.set_car_param(self.car_param_list[i])
            vehicle.destination_lane_indexes = self.destination_lanes_list[i]
            vehicle.route_type = self.route_type_list[i]

            self.car_list.append(vehicle)

            if index != car_num_total - 1:
                length = self.car_length_list[car_type_index_list[index + 1]]
                if jam_num > 1:
                    if len(num_node) > 0 and num_node[0] == car_count:
                        x = x + jam_gap + length
                        num_node.pop(0)
                    else:
                        x = x + length
                else:
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

        return [car.ID for car in self.car_list]

    def run(self, data_save=True, has_ui=True, **kwargs):
        if kwargs is None:
            kwargs = {}
        self.data_save = data_save
        """是否记录数据"""
        self.warm_up_step = kwargs.get("warm_up_step", int(5 * 60 / self.dt))
        """预热步数 [s]"""
        # self.dt = kwargs.get("dt", 0.1)
        """仿真步长 [s]"""
        if self.road_control:
            self.sim_step = kwargs.get("sim_step", int(10 * 60 / self.dt)) + 1
        else:
            self.sim_step = kwargs.get("sim_step", int(10 * 60 / self.dt))
        """总仿真步 [次]"""
        frame_rate = kwargs.get("frame_rate", -1)
        """pygame刷新率 [fps]"""
        caption = kwargs.get("ui_caption", "微观交通流仿真")
        self.yield_ = kwargs.get("if_yield", True)
        """run()是否为迭代器"""
        self.has_ui = has_ui
        """是否显示UI"""
        self.force_speed_limit = kwargs.get("force_speed_limit", False)
        """是否强制车辆速度不超过道路限速"""
        self.state_update_method = kwargs.get("state_update_method", "Euler")

        if self.has_ui and not self.road_control:
            self.ui.ui_init(caption=caption, frame_rate=frame_rate)

        # 整个仿真能够运行sim_step的仿真步
        while self.sim_step != self.step_:
            if not self.is_circle:
                self.car_summon()
            for car in self.car_list:
                car.hist_traj.append(car.get_traj_point())
                if len(car.hist_traj) > 20:
                    car.hist_traj.pop(0)
            # 能够记录warm_up_step仿真步时的车辆数据
            if self.data_save and self.step_ >= self.warm_up_step:
                self.record()
            # print("车辆生成与记录完成")
            if self.road_control: yield self.step_
            # 控制车辆对应的step需要在下一个仿真步才能显现到数据记录中
            self.update_state()  # 更新车辆状态
            # print("车辆状态更新完成")
            if self.road_control: yield self.step_
            self.step_ += 1
            self.time_ += self.dt
            if self.has_ui and not self.road_control: self.ui.ui_update()

    @abc.abstractmethod
    def update_state(self):
        pass

    def car_summon(self):
        """用于开边界车辆生成"""
        pass

    def record(self):
        for car in self.car_list:
            if car.type != V_TYPE.OBSTACLE:
                car.record()

    def _make_dummy_car(self, pos):
        car = Vehicle(self, V_TYPE.OBSTACLE, -1, 1e-5)
        car.set_cf_model(CFM.DUMMY, {})
        car.x = pos + 1e-5
        car.speed = 0
        car.acc = 0
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

    def car_insert(self, car_length: float, car_type: V_TYPE, car_class: V_CLASS, car_pos: float,
                   car_speed: float, car_acc: float,
                   cf_name: str, cf_param: dict[str, float], car_param: dict,
                   lc_name: Optional[str] = None, lc_param: Optional[dict[str, float]] = None,
                   destination_lanes=tuple[int], route_type=None):
        car = self._make_car(
            car_length, car_type, car_class, car_pos,
            car_speed, car_acc,
            cf_name, cf_param, car_param,
            lc_name, lc_param,
            destination_lanes, route_type
        )
        self.car_insert_by_instance(car)
        return car

    def car_remove(self, car: Vehicle, put_out_car_has_data=False):
        if put_out_car_has_data:
            self.out_car_has_data.append(car)
        self.car_list.remove(car)
        if hasattr(car, "plot_item"):
            car.__getattribute__("screen").removeItem(car.plot_item)
        if car.leader is not None:
            car.leader.follower = car.follower
        if car.follower is not None:
            car.follower.leader = car.leader

    def _make_car(self, car_length, car_type, car_class, car_pos,
                  car_speed, car_acc,
                  cf_name, cf_param, car_param,
                  lc_name, lc_param,
                  destination_lanes, route_type):
        VehClass = get_veh_class(car_class)
        car = VehClass(self, car_type, self._get_new_car_id(), car_length)
        car.set_cf_model(cf_name, cf_param)
        car.set_lc_model(lc_name, lc_param)
        car.set_car_param(car_param)
        car.x = car_pos
        car.speed = car_speed
        car.acc = car_acc
        car.y = self.y_center
        car.destination_lane_indexes = destination_lanes
        car.route_type = route_type
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
        return True

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
        return None

    def _get_car(self, id_):
        for car in self.car_list:
            if car.ID == id_:
                return car

    def car_insert_middle(self, car_length: float, car_type: str, car_speed: float, car_acc: float,
                          cf_name: str, cf_param: dict[str, float], car_param: dict, front_car_id: int,
                          lc_name: Optional[str] = None, lc_param: Optional[dict[str, float]] = None):
        follower_id = self.get_relative_id(front_car_id, -1)
        follower_pos = self.get_car_info(follower_id, C_Info.x)
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

    def get_relative_car(self, car: Vehicle)\
            -> tuple[Optional[Vehicle], Optional[Vehicle]]:
        """获取指定位置的前后车(用于换道)"""
        leader_ = None
        pos = car.x

        if car in self.car_list:
            return car.follower, car.leader

        for i in range(len(self.car_list)):
            temp_car = self.car_list[i]
            if temp_car.x >= pos:
                leader_ = temp_car
                break

        if leader_ is not None:
            return leader_.follower, leader_
        elif len(self.car_list) != 0:
            return self.car_list[-1], None
        else:
            return None, None

    def _common_get_relative_car(self, pos: float):
        for car in self.car_list:
            if car.x > pos:
                return car.follower, car
        if len(self.car_list) != 0:
            if self.is_circle is True:
                return self.car_list[-1], self.car_list[0]
            else:
                return self.car_list[-1], None
        return None, None

    def car_param_update(self, id_, cf_param: dict[str, float] = None, lc_param: dict[str, float] = None,
                         car_param: dict[str, float] = None):
        car = self._get_car(id_)
        car.cf_model.param_update(cf_param if cf_param is not None else {})
        car.lc_model.param_update(lc_param if lc_param is not None else {})
        car.set_car_param(car_param if car_param is not None else {})

    def __repr__(self):
        return "lane_length: " + str(self.lane_length) + \
            "\tcar_num: " + str(self.car_num_list) + \
            "\tcar_length: " + str(self.car_length_list) + \
            "\tcar_speed: " + str(self.car_initial_speed_list) + \
            "\tbasic_record: " + str(self.data_save) + \
            "\thas_ui: " + str(self.has_ui) + \
            "\tframe_rate" + str(self.ui.frame_rate) + \
            "\tdt: " + str(self.dt) + \
            "\twarm_up_step: " + str(self.warm_up_step) + \
            "\tsim_step: " + str(self.sim_step)

    def reset(self):
        """清除车道上的车辆"""
        for car in self.car_list:
            if hasattr(car, "plot_item"):
                car.__getattribute__("screen").removeItem(car.plot_item)
        self.car_list.clear()
        self._dummy_car_list.clear()
        self.out_car_has_data.clear()
        self.sim_step = 0
        self.step_ = 0
        self.time_ = 0
        self.data_container.reset()
