# -*- coding: utf-8 -*-
# @time : 2023/5/27 12:21
# @Author : yzbyx
# @File : run_basic_diagram.py
# Software: PyCharm
from trasim_simplified.core.constant import CFM
from trasim_simplified.util.flow_basic.basic_diagram import BasicDiagram


def run_basic_diagram(cf_name_, tau_, is_jam_, cf_param_, initial_v=0., random_v=False,
                      car_length_=5., start=0.01, end=1., step=0.02, plot=True, resume=False, parallel=True):
    diag = BasicDiagram(1000, car_length_, initial_v, random_v, cf_mode=cf_name_, cf_param=cf_param_)
    diag.run(start, end, step, resume=resume, file_name="result_" + cf_name_ + ("_jam" if is_jam_ else "") +
                                                        f"_{initial_v}_{random_v}",
             dt=tau_, jam=is_jam_, state_update_method="Euler", parallel=parallel)
    diag.get_by_equilibrium_state_func()
    if plot: diag.plot()


def run_all():
    for params in [
        (CFM.ACC, 0.1, False, {"original_acc": True, "tau": 0.1}, 5),
        (CFM.ACC, 1, False, {"original_acc": False, "v_safe_dispersed": True, "tau": 1}, 7.5),
        (CFM.CACC, 0.1, False, {}, 5),
        (CFM.TPACC, 1, False, {}, 7.5),
        (CFM.KK, 1, False, {}, 7.5),
        (CFM.GIPPS, 0.7, False, {}, 5),
        (CFM.IDM, 0.1, False, {}, 5),
        (CFM.OPTIMAL_VELOCITY, 0.1, False, {}, 5),
        (CFM.WIEDEMANN_99, 0.1, False, {}, 5)
    ]:
        run_basic_diagram(*params, start=0.1, end=0.91, step=0.1, plot=False)


if __name__ == '__main__':
    cf_name = CFM.ACC
    tau = 0.1
    speed = 0  # 初始速度为负代表真实的初始速度为跟驰模型期望速度
    # cf_param = {"lambda": 0.8, "original_acc": True, "v_safe_dispersed": True, "tau": tau}
    # cf_param = {"original_acc": True, "tau": 0.1, "v0": 30}
    cf_param = {"k1": 0.23, "k2": 0.5, "s0": 2, "original_acc": True, "thw": 1.6, "tau": tau, "v0": 30}
    # cf_param = {"omega": 0.8, "v0": 30}
    car_length = 5
    run_basic_diagram(cf_name, tau, False, cf_param, car_length_=car_length, initial_v=speed, resume=False,
                      parallel=True)
