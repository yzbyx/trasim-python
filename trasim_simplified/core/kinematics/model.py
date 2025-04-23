# -*- coding = uft-8 -*-
# @time : 2023-03-24 12:12
# @Author : yzbyx
# @File : model.py
# @Software : PyCharm
import abc
from typing import TYPE_CHECKING, Optional

from trasim_simplified.core.constant import VehSurr

if TYPE_CHECKING:
    from trasim_simplified.core.agent.vehicle import Vehicle


class Model(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = None
        self.thesis = None
        self.dt = None
        self.veh_surr: Optional['VehSurr'] = None

    @abc.abstractmethod
    def _update_dynamic(self):
        """
        更新step所需的动态参数

        :_param kwargs: 动态参数字典（可选）若无，则依照vehicle动态获取数据
        """
        pass

    @abc.abstractmethod
    def step(self, veh_surr: 'VehSurr'):
        """
        计算下一时间步的加速度

        :return: 下一时间步的加速度
        """
        pass

    def param_update(self, param: dict[str, float]) -> None:
        """
        更新跟驰参数

        :_param param: 包含待更新参数的字典
        """
        if param is None:
            return
        for key in param.keys():
            inner_name = "_" + key
            if hasattr(self, inner_name):
                setattr(self, inner_name, param.get(key))
            else:
                print(f"{self.name}模型无参数{key}!")

    def get_param_map(self) -> dict[str, float]:
        """获取模型参数值"""
        param_map = {}
        for name_ in self.__dict__:
            if name_[0] == "_" and name_[:2] != "__" and not callable(name_):
                param_map[name_[1:]] = getattr(self, name_)
        return param_map
