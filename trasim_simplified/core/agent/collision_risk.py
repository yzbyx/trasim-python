from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import multivariate_normal

if TYPE_CHECKING:
    from trasim_simplified.core.agent import Vehicle


# 计算目标车辆角点与前车或后车的碰撞概率
def compute_collision_probability(x_min, x_max, y_min, y_max, param):
    """
    计算目标车辆角点与前车或后车的碰撞概率
    :param x_min: 目标车辆角点的最小x坐标
    :param x_max: 目标车辆角点的最大x坐标
    :param y_min: 目标车辆角点的最小y坐标
    :param y_max: 目标车辆角点的最大y坐标
    :param center_point: 目标车辆中心点坐标
    :param param: 前车或后车的轨迹分布参数 (sigma_x, sigma_y, rho)
    """
    mu_x, mu_y, sigma_x, sigma_y, rho = param
    cov = np.array([[sigma_x ** 2, rho * sigma_x * sigma_y],
                    [rho * sigma_x * sigma_y, sigma_y ** 2]])
    multivariate_normal_pdf = multivariate_normal(mean=(mu_x, mu_y), cov=cov)

    # 蒙特卡洛方法估算概率
    num_samples = 1000
    samples = multivariate_normal_pdf.rvs(size=num_samples)
    mask = (samples[:, 0] >= x_min) & (samples[:, 0] <= x_max) & \
           (samples[:, 1] >= y_min) & (samples[:, 1] <= y_max)
    probability_mc = np.sum(mask) / num_samples

    return probability_mc


# 计算最大化的碰撞概率
def calculate_max_collision_risk(
        target_traj, target_width, target_length,
        front_traj_prob, front_width, front_length,
        rear_traj_prob, rear_width, rear_length,
        ori_front_traj, ori_front_width, ori_front_length,
):
    """
    计算目标车辆与前后车在每个时间步的角点碰撞概率，并找到最大碰撞概率
    target_traj_bbox: 目标车辆的角点轨迹 (N, 4, 2)
    front_traj: 前车的角点轨迹 (N, 2)
    param_front: 前车的轨迹分布参数 (mu_x, mu_y, sigma_x, sigma_y, rho)
    rear_traj: 后车的角点轨迹 (N, 2)
    param_rear: 后车的轨迹分布参数 (mu_x, mu_y, sigma_x, sigma_y, rho)
    """
    collision_p_list = []
    # 对每个时间步计算目标车与前后车的角点碰撞概率
    for t in range(len(target_traj)):
        target_point = target_traj[t]

        # 计算四个角点与前车和后车的碰撞概率
        x_min, x_max = target_point[0] - target_length / 2, target_point[0] + target_length / 2
        y_min, y_max = target_point[1] - target_width / 2, target_point[1] + target_width / 2

        # x_min_f = x_min - front_length / 2
        # x_max_f = x_max + front_length / 2
        # y_min_f = y_min - front_width / 2
        # y_max_f = y_max + front_width / 2
        x_min_f = x_min
        x_max_f = x_max
        y_min_f = y_min
        y_max_f = y_max
        front_point = front_traj_prob[t]
        front_prob = compute_collision_probability(x_min_f, x_max_f, y_min_f, y_max_f, front_point)

        # 计算与后车的碰撞概率
        # x_min_r = x_min - rear_length / 2
        # x_max_r = x_max + rear_length / 2
        # y_min_r = y_min - rear_width / 2
        # y_max_r = y_max + rear_width / 2
        x_min_r = x_min
        x_max_r = x_max
        y_min_r = y_min
        y_max_r = y_max
        rear_point = rear_traj_prob[t]
        rear_prob = compute_collision_probability(x_min_r, x_max_r, y_min_r, y_max_r, rear_point)

        # 计算与前车的碰撞概率
        # x_min_ori = x_min - ori_front_length / 2
        # x_max_ori = x_max + ori_front_length / 2
        # y_min_ori = y_min - ori_front_width / 2
        # y_max_ori = y_max + ori_front_width / 2
        x_min_ori = x_min
        x_max_ori = x_max
        y_min_ori = y_min
        y_max_ori = y_max
        front_ori_point = ori_front_traj[t]
        front_ori_prob = compute_collision_probability(x_min_ori, x_max_ori, y_min_ori, y_max_ori, front_ori_point)

        # 选择当前角点的最大碰撞概率
        max_prob = max(front_prob, rear_prob, front_ori_prob)
        collision_p_list.append(max_prob)

    # 计算所有时间步的最大碰撞概率
    max_collision_probability = max(collision_p_list)
    return max_collision_probability
