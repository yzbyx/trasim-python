# -*- coding: utf-8 -*-
# @Time : 2023/5/25 10:47
# @Author : yzbyx
# @File : test_basic_diagram.py
# Software: PyCharm
from trasim_simplified.core.constant import CFM
from trasim_simplified.util.flow_basic.basic_diagram import BasicDiagram


def test_basic_diagram():
    cf_name = CFM.KK
    tau = 1
    is_jam = True
    cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True, "tau": tau, "k2": 0.6}
    diag = BasicDiagram(2000, 7.5, 0, False, cf_mode=cf_name, cf_param=cf_param)
    diag.run(0.01, 1, 0.02, resume=False, file_name="result_" + cf_name + ("_jam" if is_jam else ""),
             dt=tau, jam=is_jam)
    diag.get_by_equilibrium_state_func()
    diag.plot()


if __name__ == '__main__':
    test_basic_diagram()
