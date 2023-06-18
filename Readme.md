# Trasim - Traffic Simulation

---

## Trasim是什么

---

Trasim是一个以微观连续交通流为核心的交通仿真项目，包含部分宏观连续交通流实现。

## Trasim可以做什么

----

* **静态道路场景构建**：单/多车道、开/闭环边界、车道速度/换道控制、基本段/汇入汇出段；
* **车辆模型自定义**：单种/混合的跟驰/换道模型、车辆尺寸/颜色、车流分布/大小；
* **可视化场景支持**：实时2D车辆可视化；
* **仿真条件动态控制**：任意时间步的单车/多车加速度控制与属性更新、更新道路控制策略；
* **结果分析工具**：指标结果输出、时空图绘制、指定车辆的指标关系序列绘制、基本图生成；
* ...

## 局限性

---

* 不能实现多个路段的仿真，即没有道路节点的概念，由此导致交叉口、道路网仿真无法实现

## 快速上手

---

### 0. 前置内容

Trasim作为以Python语言为基础的交通流仿真工具，使用者或贡献者需要提前了解以下内容：

* 面向对象的思想和在Python中的实现
* 基本的流程控制和函数定义语法（for、while、if、yield、def...）
* matplotlib、numpy、pandas库的基本使用

### 1. 多车道+匝道汇入+单车扰动仿真示例

```python
from trasim_simplified.core.constant import V_TYPE, CFM, COLOR, LCM, SECTION_TYPE
from trasim_simplified.core.data.data_plot import Plot
from trasim_simplified.core.frame.micro.open_lane import THW_DISTRI
from trasim_simplified.core.frame.micro.road import Road
from trasim_simplified.util.decorator.timer import timer_no_log
from trasim_simplified.core.data.data_container import Info as C_Info


@timer_no_log
def run_road():
    _cf_param = {"lambda": 0.8, "original_acc": True, "v_safe_dispersed": True}
    """跟驰模型参数"""
    _car_param = {}
    """车辆物理参数：尺寸、颜色等"""
    take_over_index = -1
    """受控车辆ID"""
    follower_index = -1
    """受控车辆的跟驰车辆ID"""
    dt = 1
    """仿真步长 [s]"""
    warm_up_step = 0
    """预热时长，之后进行车辆数据记录 [s]"""
    sim_step = warm_up_step + int(1200 / dt)
    """总仿真时长 [s]"""
    offset_step = int(300 / dt)
    """车辆控制偏移时长（避免控制效果出现在数据记录的开始） [s]"""

    is_circle = False
    """是否为循环边界"""
    road_length = 10000
    """道路长度 [m]"""
    lane_num = 3
    """车道数量"""

    if is_circle:
        sim = Road(road_length)
        # 新建长度为road_length的道路对象
        lanes = sim.add_lanes(lane_num, is_circle=True)
        # 新建指定数量的车道，并获取车道对象列表
        for i in range(lane_num):
            # 配置每条车道的车辆：数量、车身长度、车辆类型、车辆初始速度、是否添加随机速度扰动、跟驰模型
            # 跟驰模型参数、车辆属性、换道模型、换道模型参数
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.yellow},
                                lc_name=LCM.KK, lc_param={})
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.blue},
                                lc_name=LCM.KK, lc_param={})
            # 加载车辆到车道上
            lanes[i].car_load()
            # 数据记录配置初始化
            lanes[i].data_container.config()
    else:
        sim = Road(road_length)
        lanes = sim.add_lanes(lane_num, is_circle=False)
        for i in range(lane_num):
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, 20, False, CFM.KK, _cf_param, {"color": COLOR.yellow},
                                lc_name=LCM.KK, lc_param={})
            lanes[i].car_config(200, 7.5, V_TYPE.PASSENGER, 20, False, CFM.TPACC, _cf_param, {"color": COLOR.blue},
                                lc_name=LCM.ACC, lc_param={})
            lanes[i].data_container.config()

            if i != lane_num - 1:
                # 配置车道的流入流量以及车头时距分布，
                # 根据car_config的车辆数量按比例随机生成不同类型的车辆
                lanes[i].car_loader(2000, THW_DISTRI.Uniform)
            else:
                lanes[i].car_loader(200, THW_DISTRI.Uniform)

            if i == lane_num - 2:
                # 设置车道指定范围的类型，用于跟驰和换道模型内部的处理
                lanes[i].set_section_type(SECTION_TYPE.BASE)
                lanes[i].set_section_type(SECTION_TYPE.NO_RIGHT)
            if i == lane_num - 1:
                lanes[i].set_section_type(SECTION_TYPE.ON_RAMP, 10000, -1)
                lanes[i].set_section_type(SECTION_TYPE.NO_LEFT, 0, 10000)
                lanes[i].set_section_type(SECTION_TYPE.BASE, 0, 10000)
                # 在指定位置设置路障
                lanes[i].set_block(10300)
            else:
                pass

    # 仿真主循环，step代表此时的仿真步，
    # state代表单个循环的状态（0代表跟驰计算完成、1代表换道计算完成）
    for step, state in sim.run(data_save=True, has_ui=False, frame_rate=-1,
                               warm_up_step=warm_up_step, sim_step=sim_step, dt=dt):
        # 当跟驰计算完成后获取距离道路中心最近的车辆ID作为受控车辆
        if warm_up_step + offset_step == step and state == 0:
            take_over_index = sim.get_appropriate_car(lane_add_num=0)
            print(take_over_index)
        # 接管受控车辆，使其以指定的加速度和换道选择进行运动
        if warm_up_step + offset_step <= step < warm_up_step + offset_step + int(60 / dt):
            sim.take_over(take_over_index, -3, lc_result={"lc": 0})

    # 获取记录的车辆指标数据，格式为pandas.DataFrame
    df = sim.data_to_df()
    # 获取受控车辆经过的车道ID列表
    lane_ids = sim.find_on_lanes(take_over_index)
    # 绘制受控车辆在单一车道上的基本指标关系的时间序列曲线
    Plot.basic_plot(take_over_index, lane_id=lane_ids[0], data_df=df)
    # 绘制指定车道的时空图，指定控制轨迹颜色的指标，并高亮受控车辆的轨迹
    Plot.spatial_time_plot(take_over_index, lane_add_num=0,
                           color_info_name=C_Info.v, data_df=df, single_plot=False)
    Plot.spatial_time_plot(take_over_index, lane_add_num=1,
                           color_info_name=C_Info.v, data_df=df, single_plot=False)
    Plot.spatial_time_plot(take_over_index, lane_add_num=2,
                           color_info_name=C_Info.v, data_df=df, single_plot=False)
    # 显示绘图结果
    Plot.show()

    # 设置虚拟检测范围，获取集计指标结果
    result = sim.data_processor.aggregate_as_detect_loop(
        df, 0, road_length, 0, 995, dt, 300, [warm_up_step, sim_step]
    )
    # 打印输出结果
    sim.data_processor.print(result)


if __name__ == '__main__':
    run_road()
```

### 2. 基本图仿真绘制

```python
from trasim_simplified.core.constant import CFM
from trasim_simplified.util.flow_basic.basic_diagram import BasicDiagram

def run_basic_diagram(cf_name_, tau_, is_jam_, cf_param_, initial_v=0., random_v=False,
                      car_length_=5., start=0.01, end=1., step=0.02, plot=True, resume=False):
    diag = BasicDiagram(1000, car_length_, initial_v, random_v, cf_mode=cf_name_, cf_param=cf_param_)
    diag.run(start, end, step, resume=resume, file_name="result_" + cf_name_ + ("_jam" if is_jam_ else "") +
                                                        f"_{initial_v}_{random_v}",
             dt=tau_, jam=is_jam_, state_update_method="Euler")
    diag.get_by_equilibrium_state_func()
    if plot: diag.plot()


if __name__ == '__main__':
    cf_name = CFM.KK
    """跟驰模型名称"""
    tau = 1
    """仿真步长 [s]"""
    speed = 0  # 初始速度为负代表真实的初始速度为跟驰模型期望速度
    """初始速度 [m/s]"""
    cf_param = {"lambda": 0.8, "original_acc": False, "v_safe_dispersed": True, "tau": tau, "k2": 0.3}
    """跟驰模型参数"""
    car_length = 7.5
    """车辆长度 [m]"""
    # 运行基本图程序，参数包括：跟驰模型、仿真步长、初始是否为阻塞状态、跟驰模型参数、初始速度、是否添加初始随机速度扰动、
    # 车辆长度、开始Occ、结束Occ、步进Occ、是否绘图、是否接续之前的结果继续运行
    run_basic_diagram(cf_name, tau, False, cf_param, car_length_=car_length, initial_v=speed, resume=False)
```

## 说明文档

---

