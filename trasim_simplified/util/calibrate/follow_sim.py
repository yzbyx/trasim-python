# -*- coding: utf-8 -*-
# @Time : 2023/10/14 21:25
# @Author : yzbyx
# @File : follow_sim.py
# Software: PyCharm
from trasim_simplified.msg.trasimError import TrasimError


def simulation(cf_func, obs_v, obs_x, obs_lv, obs_lx, cf_params, dt,
               leaderL: float, update_method="Euler") -> tuple[list, list, list]:
    """
    给定前车的位置、速度，以及模型参数，仿真得到后车的位置、速度、加速度，默认前车ID不变

    注意！原始数据的位置、速度的关系需要与模型状态的更新方式相对应
    """
    tau = cf_params.get('tau', False)
    if tau:
        assert tau >= dt
    speed = obs_v[0]
    pos = obs_x[0]
    sim_pos, sim_speed, sim_acc = [pos], [speed], [0]
    for i, x, v, lx, lv in enumerate(zip(obs_x, obs_v, obs_lx, obs_lv)):
        cf_acc = cf_func(**cf_params, speed=speed, gap=lx - x - leaderL, leaderV=obs_lv, interval=dt)
        if update_method == "Euler":  # 差分型，最常用
            speed += cf_acc * dt
            pos += speed * dt
        elif update_method == "Ballistic":  # Treiber et al., 2006; Treiber and Kanagaraj, 2015 速度渐变型
            speed_before = speed
            speed += cf_acc * dt
            pos += (speed + speed_before) * dt / 2
        else:
            raise TrasimError("update_method must be 'Euler' or 'Ballistic'")
        sim_pos.append(pos)
        sim_speed.append(speed)
        sim_acc.append(cf_acc)
    return sim_pos, sim_speed, sim_acc
