# -*- coding: utf-8 -*-
# @time : 2025/3/22 20:53
# @Author : yzbyx
# @File : __init__.py.py
# Software: PyCharm
from trasim_simplified.core.agent.vehicle import Vehicle
from trasim_simplified.core.agent.game_agent import Game_A_Vehicle, Game_H_Vehicle, Game_O_Vehicle
from trasim_simplified.core.constant import V_CLASS


def get_veh_class(veh_class: V_CLASS):
    """根据车辆类别返回车辆类型"""
    if veh_class == V_CLASS.BASE:
        return Vehicle
    elif veh_class == V_CLASS.GAME_AV:
        return Game_A_Vehicle
    elif veh_class == V_CLASS.GAME_HV:
        return Game_H_Vehicle
    elif veh_class == V_CLASS.GAME_OV:
        return Game_O_Vehicle
    else:
        raise ValueError(f"Unknown vehicle class: {veh_class}")
