# -*- coding = uft-8 -*-
# @Time : 2022-02-26 9:49
# @Author : yzbyx
# @File : timer.py
# @Software : PyCharm

import time
import warnings
from functools import wraps


def _get_current_time():
    t = time.time()
    data_head = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    time_stamp = "%s.%s" % (data_head, str(t).split('.')[-1][:5])
    return time_stamp


def timer(logfile='my_out_time.log'):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            timeIn = time.time()
            timeStart = _get_current_time()
            result = func(*args, **kwargs)
            timeOut = time.time()
            log_string = '[' + func.__name__ + '] ' + 'time usage: ' + timeStart + ' + ' + \
                         str((timeOut - timeIn) * 1000 // 1 / 1000) + ' s'
            # 打开logfile，并写入内容
            with open(logfile, 'a') as opened_file:
                # 现在将日志打到指定的logfile
                opened_file.write(log_string + '\n')
            return result

        return wrapped_function

    return logging_decorator


def timer_no_log(func):
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        timeIn = time.time()
        timeStart = _get_current_time()
        result = func(*args, **kwargs)
        timeOut = time.time()
        log_string = '[' + func.__name__ + '] ' + 'time usage: ' + timeStart + ' + ' + \
                     str((timeOut - timeIn) * 1000 // 1 / 1000) + ' s'
        print(log_string)
        return result

    return wrapped_function


def deprecated(newFuncName=None):
    """过时函数装饰器"""
    def Inner(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if newFuncName is None:
                msg = f"函数{func.__name__}即将过时！"
            else:
                msg = f"函数{func.__name__}即将过时，请使用{newFuncName}代替！"
            warnings.warn(msg, stacklevel=2)
            return func(*args, **kwargs)
        return wrapped_func
    return Inner


@timer()
def test():
    s = 0
    for i in range(10000000):
        s *= (i + 1)


if __name__ == '__main__':
    test()
