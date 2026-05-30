import matplotlib.pyplot as plt
import numpy as np


import numpy as np

class BicycleModelV2:
    """
    运动学自行车模型 V2:
    - 控制量: 加速度 a, 前轮转角 delta
    - (x, y) 状态代表车头(前轮轴)位置
    - 内部基于后轮轴进行状态更新
    """
    def __init__(self, x_front=0, y_front=0, theta=0, v=5, L=2.5, dt=0.1):
        self.L = L          # 轴距 (m)
        self.dt = dt        # 时间步长 (s)
        self.theta = theta  # 朝向角 (rad)
        self.v = v          # 速度 (m/s)
        self.f = L / 3      # 前轮轴到车头的距离 (m)

        # 内部状态：后轮轴位置
        self.x_rear = x_front - self.L * np.cos(self.theta)
        self.y_rear = y_front - self.L * np.sin(self.theta)

    def update(self, a, delta):
        # 限制输入
        delta = np.clip(delta, -np.pi/4, np.pi/4) # 限制最大转角
        a = np.clip(a, -3, 2) # 限制最大加/减速度 m/s^2

        # 1. 更新速度
        self.v += a * self.dt
        self.v = max(0, self.v) # 速度不能为负

        # 2. 更新后轮轴位置和朝向
        # 使用当前时间步的平均速度可以提高数值稳定性
        v_avg = self.v - 0.5 * a * self.dt
        self.x_rear += v_avg * np.cos(self.theta) * self.dt
        self.y_rear += v_avg * np.sin(self.theta) * self.dt
        self.theta += v_avg * np.tan(delta) / self.L * self.dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def get_front_axle_state(self):
        """返回车头(前轮轴)的状态"""
        x_front = self.x_rear + (self.L + self.f) * np.cos(self.theta)
        y_front = self.y_rear + (self.L + self.f) * np.sin(self.theta)
        return x_front, y_front, self.theta, self.v

from enum import Enum

class LaneChangeState(Enum):
    ACCELERATING = 1
    CHANGING_LANE = 2
    DECELERATING = 3
    FINISHED = 4

class AdvancedLaneChange:
    def __init__(self, bike, target_y, lane_change_dist,
                 initial_speed, lane_change_speed, final_speed,
                 Kp_accel=0.8, Kp_decel=0.8):
        self.bike: BicycleModelV2 = bike
        self.state = LaneChangeState.ACCELERATING

        # 速度参数
        self.initial_speed = initial_speed
        self.lane_change_speed = lane_change_speed
        self.final_speed = final_speed

        # 轨迹参数
        self.start_y = None # 将在换道开始时记录
        self.target_y = target_y
        self.lane_width = None
        self.lane_change_dist = lane_change_dist
        self.start_x = None # 将在换道开始时记录

        # P控制器增益
        self.Kp_accel = Kp_accel
        self.Kp_decel = Kp_decel

    def get_ref_y(self, x):
        """计算正弦参考轨迹的y值"""
        x_rel = x - self.start_x
        if not (0 <= x_rel <= self.lane_change_dist):
            return self.target_y if x_rel > self.lane_change_dist else self.start_y

        y_ref = self.start_y + (self.lane_width / 2.0) * \
                (1 - np.cos(np.pi * x_rel / self.lane_change_dist))
        return y_ref

    def control(self):
        x, y, theta, v = self.bike.x_rear, self.bike.y_rear, self.bike.theta, self.bike.v

        # --- 状态机逻辑 ---
        if self.state == LaneChangeState.ACCELERATING:
            # 目标: 加速到换道速度
            delta = 0.0
            accel = self.Kp_accel * (self.lane_change_speed - v)

            # 平滑jerk值

            # 状态切换条件
            if v >= self.lane_change_speed - 0.1: # 留一点容差
                self.state = LaneChangeState.CHANGING_LANE
                self.start_x = x
                self.start_y = y
                self.lane_width = self.target_y - self.start_y
                print("State -> CHANGING_LANE")

        elif self.state == LaneChangeState.CHANGING_LANE:
            # 目标: 沿正弦轨迹换道
            accel = 0.0 # 保持匀速

            # Pure Pursuit 控制器计算 delta
            lookahead_dist = max(3.0, v * 0.5) # 前视距离随速度变化
            x_lookahead = x + lookahead_dist * np.cos(theta)
            y_ref_lookahead = self.get_ref_y(x_lookahead)

            alpha = np.arctan2(y_ref_lookahead - y, v * self.bike.dt) - theta
            delta = np.arctan2(2 * self.bike.L * np.sin(alpha), lookahead_dist)

            # 状态切换条件
            if (x - self.start_x) >= self.lane_change_dist:
                self.state = LaneChangeState.DECELERATING
                print("State -> DECELERATING")

        elif self.state == LaneChangeState.DECELERATING:
            # 目标: 减速到最终速度
            delta = 0.0
            accel = self.Kp_decel * (self.final_speed - v)

            # 状态切换条件
            if v <= self.final_speed + 0.1:
                self.state = LaneChangeState.FINISHED
                accel = 0 # 防止过冲
                print("State -> FINISHED")

        elif self.state == LaneChangeState.FINISHED:
            accel = 0.0
            delta = 0.0

        else:
            raise ValueError(f"Unknown state: {self.state}")

        return accel, delta


# --- 仿真设置 ---
INITIAL_SPEED = 5     # 初始速度 5 m/s (18 km/h)
LANE_CHANGE_SPEED = 10  # 换道速度 10 m/s (36 km/h)
FINAL_SPEED = 7       # 最终速度 7 m/s (25.2 km/h)
LANE_WIDTH = 3.5      # 车道宽度
LANE_CHANGE_DIST = 40 # 换道所需纵向距离

# --- 初始化 ---
bike = BicycleModelV2(x_front=0, y_front=0, v=INITIAL_SPEED, dt=0.1)
controller = AdvancedLaneChange(
    bike=bike,
    target_y=LANE_WIDTH,
    lane_change_dist=LANE_CHANGE_DIST,
    initial_speed=INITIAL_SPEED,
    lane_change_speed=LANE_CHANGE_SPEED,
    final_speed=FINAL_SPEED
)

# --- 存储历史数据 ---
history = {
    't': [], 'x': [], 'y': [], 'theta': [], 'v': [], 'a': [], 'delta': []
}
time = 0
simulation_time = 12 # 秒

# --- 仿真循环 ---
for i in range(int(simulation_time / bike.dt)):
    if controller.state == LaneChangeState.FINISHED:
        a_cmd, delta_cmd = 0, 0
    else:
        a_cmd, delta_cmd = controller.control()

    bike.update(a_cmd, delta_cmd)

    x, y, theta, v = bike.get_front_axle_state()
    history['t'].append(time)
    history['x'].append(x)
    history['y'].append(y)
    history['a'].append(v)
    history['yaw'].append(v)

    time += bike.dt

# --- 结果可视化 ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10))
fig.tight_layout(pad=4.0)

# 1. XY 轨迹
axs[0].plot(history['x'], history['y'], label='Vehicle Trajectory (Front Axle)')
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].axhline(LANE_WIDTH, color='gray', linestyle='--')
axs[0].set_title('XY Trajectory')
axs[0].set_xlabel('X Position (m)')
axs[0].set_ylabel('Y Position (m)')
axs[0].grid(True)
axs[0].legend()
axs[0].axis('equal')

# 2. 速度 vs. X距离
axs[1].plot(history['x'], history['v'])
axs[1].axhline(LANE_CHANGE_SPEED, color='r', linestyle='--', label=f'Lane Change Speed ({LANE_CHANGE_SPEED} m/s)')
axs[1].axhline(FINAL_SPEED, color='g', linestyle='--', label=f'Final Speed ({FINAL_SPEED} m/s)')
axs[1].set_title('Speed vs. Distance')
axs[1].set_xlabel('X Position (m)')
axs[1].set_ylabel('Speed (m/s)')
axs[1].grid(True)
axs[1].legend()

# 3. 控制量 vs. 时间
ax2_twin = axs[2].twinx()
p1, = axs[2].plot(history['t'], history['a'], 'b-', label='Acceleration (m/s^2)')
p2, = ax2_twin.plot(history['t'], np.rad2deg(history['delta']), 'r-', label='Steer Angle (deg)')
axs[2].set_title('Control Commands vs. Time')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Acceleration (m/s^2)', color='b')
ax2_twin.set_ylabel('Steer Angle (deg)', color='r')
axs[2].grid(True)
axs[2].legend(handles=[p1, p2])

plt.show()