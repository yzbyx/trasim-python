# -*- coding: utf-8 -*-
# @Time : 2023/10/14 21:27
# @Author : yzbyx
# @File : tools.py
# Software: PyCharm
import pickle


def open_pickle(file: str):
    with open(file, 'rb') as f:
        return pickle.load(f)
