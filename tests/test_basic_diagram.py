# -*- coding: utf-8 -*-
# @Time : 2023/5/25 10:47
# @Author : yzbyx
# @File : test_basic_diagram.py
# Software: PyCharm
import pytest

from tests.run_basic_diagram import run_basic_diagram
from trasim_simplified.core.constant import CFM


@pytest.mark.parametrize(
    "cf_name, tau, is_jam, cf_param, car_length",
    [
        (CFM.ACC, 0.1, False, {"original_acc": True, "tau": 0.1}, 5),
        (CFM.ACC, 1, False, {"original_acc": False, "v_safe_dispersed": True, "tau": 1}, 7.5),
        (CFM.CACC, 0.1, False, {}, 5),
        (CFM.TPACC, 1, False, {}, 7.5),
        (CFM.KK, 1, False, {}, 7.5),
        (CFM.GIPPS, 0.7, False, {}, 5),
        (CFM.IDM, 0.1, False, {}, 5),
        (CFM.OPTIMAL_VELOCITY, 0.1, False, {}, 5),
        (CFM.WIEDEMANN_99, 0.1, False, {}, 5)
    ],
)
def test_basic_diagram(cf_name, tau, is_jam, cf_param, car_length):
    run_basic_diagram(cf_name, tau, is_jam, cf_param, car_length, start=0.1, end=0.91, step=0.2, plot=False)
