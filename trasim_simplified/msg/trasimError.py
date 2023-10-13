# -*- coding = uft-8 -*-
# @Time : 2022-04-04 17:10
# @Author : yzbyx
# @File : trasimError.py
# @Software : PyCharm

class TrasimError(ValueError):
    pass


class ErrorMessage:
    SPEED_FASTER_THAN_LIGHT = '车辆{}的车速{:.2f}m/s已超越光速！\t'
    HEADWAY_LESS_THAN_ZERO = '车辆{}与前车的车头间距{:.2f}m < 0！\t'
    CAR_COLLISION = '车辆{}与前车发生冲突！\t'
    NOT_FIND_OBJ = '未找到{}！\t'
    CANNOT_PLACE_CAR = '无法在此位置放置车辆！{}\t'

    NO_MODEL = '{}模块未创建！\t'

    OBJ_TYPE_ERROR = '{}类型和{}类型无法比较！\t'

