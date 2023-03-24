# -*- coding = uft-8 -*-
# @Time : 2023-03-24 12:12
# @Author : yzbyx
# @File : model.py
# @Software : PyCharm
import abc
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trasim_simplified.core.vehicle import Vehicle


class Model(metaclass=abc.ABCMeta):
    DEFAULT_PARAM = {}
    """模型默认参数字典"""
    NAME = None
    """模型名称"""
    THESIS = None
    """模型论文"""

    def __init__(self, vehicle: Optional['Vehicle']):
        self.vehicle: Optional['Vehicle'] = vehicle if vehicle else None
        self._param = {}

    @abc.abstractmethod
    def _update_dynamic(self):
        """
        更新step所需的动态参数

        :_param kwargs: 动态参数字典（可选）若无，则依照vehicle动态获取数据
        """
        pass

    @abc.abstractmethod
    def step(self, *args):
        """
        仿真程序内部调用函数，调用_calculate函数计算加速度

        :return: {'xOffset', 'speed', 'acc', ...}
        """
        pass

    @abc.abstractmethod
    def calculate(*args):
        pass

    @abc.abstractmethod
    def _param_update(self, param: dict[str, float]) -> None:
        """
        更新跟驰参数

        :_param param: 包含待更新参数的字典
        """
        pass

    def get_param_map(self) -> dict[str, float]:
        """获取模型参数值"""
        param_map = {}
        for name_ in globals():
            if name_[0] == "_" and name_[1] != "_" and not callable(name_):
                param_map[name_] = getattr(self, name_)
        return param_map
