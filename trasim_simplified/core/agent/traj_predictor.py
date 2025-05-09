# -*- coding: utf-8 -*-
# @time : 2025/4/5 17:00
# @Author : yzbyx
# @File : traj_predictor.py
# Software: PyCharm
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

from traj_predictor.dl_dataset import collate_fn
from traj_predictor.dl_model import PredictionNet
from traj_predictor.parser import ArgsConfig
from traj_predictor.util import load_checkpoint
from traj_process.util.plot_helper import get_fig_ax
from trasim_simplified.core.constant import VehSurr, TrajPoint

if TYPE_CHECKING:
    from trasim_simplified.core.agent import Vehicle

pred_net = None


def get_pred_net():
    """
    获取轨迹预测网络
    :return: 轨迹预测网络
    """
    global pred_net
    if pred_net is None:
        pred_net = TrajPred()
    return pred_net


def rotate(theta, arr):
    """
    逆时针旋转
    :param arr: 坐标
    :param theta: 旋转角度
    :return: 旋转后的坐标
    """
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(rot_matrix, arr)


def add_v_a_yaw(traj_list: list[TrajPoint], dt=0.1):
    """
    计算车辆的速度、加速度和航向角
    :param traj_list: 轨迹点列表
    :param dt: 时间间隔
    """
    x = np.array([traj.x for traj in traj_list])
    y = np.array([traj.y for traj in traj_list])
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    yaw = np.arctan2(vy, vx)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    acc = np.gradient(speed, dt)

    # 赋值回去
    for i, traj in enumerate(traj_list):
        traj.speed = speed[i]
        traj.yaw = yaw[i]
        traj.acc = acc[i]


class TrajPred:
    def __init__(self):
        self.net: Optional[PredictionNet] = None

    @staticmethod
    def get_scene(veh_surr: VehSurr):
        ev = veh_surr.ev
        cp = veh_surr.cp
        cr = veh_surr.cr
        lp = veh_surr.lp
        lr = veh_surr.lr
        rp = veh_surr.rp
        rr = veh_surr.rr

        # 获取周边车辆历史2s的轨迹
        traj_point = ev.get_traj_point()
        ev_pos = np.array([traj_point.x, traj_point.y])
        ev_yaw = traj_point.yaw
        dt = ev.lane.dt

        traj_hist_mask_all = []
        veh_surr_history_all = []
        for i, veh in enumerate([ev, cp, cr, lp, lr, rp, rr]):
            if veh is None:
                traj_hist_mask_all.append([1] * round(2 / dt))
                veh_surr_history_all.append([[0] * 5 + [i] for _ in range(round(2 / dt))])
                continue
            traj_s = veh.get_history_trajectory(2)
            traj_hist_mask = []
            veh_surr_history = []
            for traj in traj_s:
                if traj is None:
                    traj_hist_mask.append(1)
                    veh_surr_history.append([0] * 5 + [i])
                    continue
                else:
                    traj_hist_mask.append(0)
                # "rel_x", "rel_y", "rel_dx", "rel_dy", "theta", "idx"
                rel_pos = np.array([traj.x - ev_pos[0], traj.y - ev_pos[1]])
                # 旋转
                rel_pos = rotate(-ev_yaw, rel_pos)
                rel_dpos = np.array([traj.vx * dt, traj.vy * dt])
                # 旋转
                rel_dpos = rotate(-ev_yaw, rel_dpos)
                veh_surr_history.append(
                    [*rel_pos, *rel_dpos, veh.yaw - ev_yaw, i]
                )
            traj_hist_mask_all.append(traj_hist_mask)
            veh_surr_history_all.append(veh_surr_history)

        traj_hist_mask_all = np.array(traj_hist_mask_all)
        veh_surr_history_all = np.array(veh_surr_history_all)

        # 获取当前车道、左车道、右车道的前后100m中心线
        lane_data_all = []
        lane_data_mask_all = []
        c_lane = ev.lane
        l_lane = ev.left_lane
        r_lane = ev.right_lane
        for i, lane in enumerate([c_lane, l_lane, r_lane]):
            if lane is None:
                lane_data_all.append([[0] * 5 + [i] for _ in range(100)])
                lane_data_mask_all.append([1] * 100)
            else:
                # 获取车道中心线
                center_pos_array = np.array([(- 49 + i, lane.y_center - ev.y) for i in range(100)])
                # 旋转
                center_pos_array = rotate(-ev_yaw, center_pos_array.T).T
                center_dpos_array = np.array([(1, 0) for _ in range(100)])
                # 旋转
                center_dpos_array = rotate(-ev_yaw, center_dpos_array.T).T

                yaw_array = np.array([-ev_yaw for _ in range(100)])
                # 横向拼接
                lane_data_all.append(
                    np.concatenate(
                        (center_pos_array, center_dpos_array,
                         yaw_array.reshape(-1, 1), np.array([i] * 100).reshape(-1, 1)), axis=1
                    )
                )
                lane_data_mask_all.append([0] * 100)

            # lane_data_all.append(lane_data)
            # lane_data_mask_all.append(lane_data_mask)

        lane_data_all = np.array(lane_data_all)
        lane_data_mask_all = np.array(lane_data_mask_all)

        return {
            'traj_hist': veh_surr_history_all,
            'traj_hist_mask': traj_hist_mask_all,
            'lane_string_data': lane_data_all,
            'lane_data_mask': lane_data_mask_all
        }

    @staticmethod
    def plot_scene(scene_dict):
        plt.ioff()
        traj_hist = scene_dict['traj_hist']
        traj_hist_mask = scene_dict['traj_hist_mask']
        lane_string_data = scene_dict['lane_string_data']
        lane_data_mask = scene_dict['lane_data_mask']
        fig, ax = get_fig_ax()
        # 绘制历史轨迹
        for i, traj in enumerate(traj_hist):
            traj = traj[traj_hist_mask[i] == 0]
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], color="red" if i == 0 else 'b')
        # 绘制车道线
        for i, lane in enumerate(lane_string_data):
            lane = lane[lane_data_mask[i] == 0]
            lane = np.array(lane)
            ax.plot(lane[:, 0], lane[:, 1], color="gray", alpha=0.2)
        plt.show()
        plt.ion()

    def pred_traj(self, veh_surr: VehSurr, type_="net", time_len=3):
        """
        预测轨迹，包括当前轨迹点，轨迹时长为time_len+dt
        :param veh_surr:
        :param type_:
        :param time_len:
        :return: 得到车头中点的轨迹
        """
        if type_ == "net":
            if veh_surr.ev.pred_traj is None:
                if self.net is None:
                    arg = ArgsConfig()
                    arg.device = "cpu"
                    arg.mode = "test"
                    arg.batch_size = 1
                    self.net = PredictionNet(arg).to(arg.device)
                    optimizer = torch.optim.Adam(self.net.parameters(), lr=arg.lr)
                    load_checkpoint(r'E:\BaiduSyncdisk\traj-predictor\model', self.net, optimizer)
                    self.net.eval()

                # 1. 生成模型输入
                input_data = self.get_scene(veh_surr)
                # self.plot_scene(input_data)
                # 2. 进行轨迹预测
                input_data_fn = collate_fn([input_data])
                predicted_trajectory = self.net(input_data_fn)
                # 3. 进行后处理
                predicted_trajectory = predicted_trajectory.cpu().detach().numpy()[0]
                yaw = veh_surr.ev.yaw
                predicted_trajectory = rotate(yaw, predicted_trajectory.T).T
                predicted_trajectory[:, 0] += veh_surr.ev.x
                predicted_trajectory[:, 1] += veh_surr.ev.y

                vx = np.gradient(predicted_trajectory[:, 0], veh_surr.ev.lane.dt)
                vy = np.gradient(predicted_trajectory[:, 1], veh_surr.ev.lane.dt)
                speed = np.sqrt(vx ** 2 + vy ** 2)
                # 计算加速度
                acc = np.gradient(speed, veh_surr.ev.lane.dt)
                # 计算航向角
                yaw = np.arctan2(vy, vx)

                pred_traj = [veh_surr.ev.get_traj_point()]
                for i in range(predicted_trajectory.shape[0]):
                    pred_traj.append(
                        TrajPoint(
                            x=predicted_trajectory[i, 0],
                            y=predicted_trajectory[i, 1],
                            speed=speed[i],
                            acc=acc[i],
                            yaw=yaw[i],
                            length=veh_surr.ev.length,
                            width=veh_surr.ev.width
                        )
                    )
                veh_surr.ev.pred_traj = pred_traj

                if time_len > 3:
                    dt = veh_surr.ev.lane.dt
                    for i in range(round((time_len - 3) / dt)):
                        pred_traj.append(
                            TrajPoint(
                                x=pred_traj[-1].x + pred_traj[-1].vx * dt,
                                y=pred_traj[-1].y + pred_traj[-1].vy * dt,
                                speed=pred_traj[-1].speed,
                                acc=0,
                                yaw=pred_traj[-1].yaw,
                                length=veh_surr.ev.length,
                                width=veh_surr.ev.width
                            )
                        )

                # 获取前time_len秒的轨迹
                add_v_a_yaw(veh_surr.ev.pred_traj, veh_surr.ev.lane.dt)
            traj_pred = veh_surr.ev.pred_traj[:round(time_len / veh_surr.ev.lane.dt) + 1]

        elif type_ == "const":
            if veh_surr.ev.pred_traj is None:
                veh_surr.ev.pred_traj = self.pred_traj_const(veh_surr.ev, 10)
                add_v_a_yaw(veh_surr.ev.pred_traj, veh_surr.ev.lane.dt)
            traj_pred = veh_surr.ev.pred_traj[:round(time_len / veh_surr.ev.lane.dt) + 1]
        else:
            raise ValueError("Invalid type_ value. Expected 'net' or 'const'.")

        return [point.copy() for point in traj_pred]

    @staticmethod
    def pred_traj_const(veh: 'Vehicle', time_len=3):
        """
        预测轨迹
        :param veh:
        :param time_len: 预测时间长度
        :return: 预测轨迹
        """
        pred_traj = [veh.get_traj_point()]
        dt = veh.lane.dt
        for i in range(round(time_len / dt)):
            pred_traj.append(
                TrajPoint(
                    x=veh.x + veh.v * dt,
                    y=veh.y + veh.v_lat * dt,
                    length=veh.length,
                    width=veh.width
                )
            )
        return pred_traj
