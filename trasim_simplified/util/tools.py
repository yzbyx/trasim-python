# -*- coding: utf-8 -*-
# @Time : 2023/10/14 21:27
# @Author : yzbyx
# @File : tools.py
# Software: PyCharm
import pickle


def save_to_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
