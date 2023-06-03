# -*- coding: utf-8 -*-
# @Time : 2023/5/27 12:21
# @Author : yzbyx
# @File : run_basic_diagram.py
# Software: PyCharm
from trasim_simplified.core.constant import CFM
from trasim_simplified.util.flow_basic.basic_diagram import BasicDiagram


def run_basic_diagram(cf_name_, tau_, is_jam_, cf_param_, car_length_=5., start=0.01, end=1., step=0.02, plot=True):
    diag = BasicDiagram(10000, car_length_, 0, False, cf_mode=cf_name_, cf_param=cf_param_)
    diag.run(start, end, step, resume=True, file_name="result_" + cf_name_ + ("_jam" if is_jam_ else ""),
             dt=tau_, jam=is_jam_, state_update_method="Euler")
    diag.get_by_equilibrium_state_func()
    if plot: diag.plot()


if __name__ == '__main__':
    cf_name = CFM.ACC
    tau = 1
    is_jam = False
    cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True, "tau": tau, "k2": 0.6}
    car_length = 7.5
    run_basic_diagram(cf_name, tau, is_jam, cf_param, car_length)
