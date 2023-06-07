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

### 1. 静态道路场景构建

[run_road.py](tests%2Frun_road.py)

## 说明文档

---
