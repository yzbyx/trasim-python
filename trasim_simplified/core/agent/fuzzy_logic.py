# -*- coding: utf-8 -*-
# @Time : 2025/3/30 23:19
# @Author : yzbyx
# @File : fuzzy_logic.py
# Software: PyCharm
import numpy as np
import scienceplots  # type: ignore
import seaborn as sns
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from matplotlib import pyplot as plt

sns.reset_orig()
plt.style.use(['science', 'no-latex', 'cjk-sc-font'])


class FuzzyLogic:
    def __init__(self):
        universe = np.arange(0, 1.01, 0.01)
        # 定义模糊变量
        self.B = ctrl.Antecedent(universe, '收益')
        self.R = ctrl.Antecedent(universe, '风险')
        self.A = ctrl.Antecedent(universe, '激进度')
        self.D = ctrl.Consequent(universe, '决策')

        # 隶属度函数
        abc_element = [
            [0, 0, 0.25], [0, 0.25, 0.5], [0.25, 0.5, 0.75], [0.5, 0.75, 1], [0.75, 1, 1]
        ]
        fuzzy_names = ["低", "较低", "中", "较高", "高"]
        fuzzy_set = list(zip(fuzzy_names, abc_element))
        for name, element in fuzzy_set:
            self.B[name] = fuzz.trimf(self.B.universe, element)
            self.R[name] = fuzz.trimf(self.R.universe, element)
            self.D[name] = fuzz.trimf(self.D.universe, element)

        agg_abc_element = [
            [0, 0, 0.5], [0, 0.5, 1], [0.5, 1, 1]
        ]
        agg_fuzzy_names = ["保守", "中性", "激进"]
        agg_fuzzy_set = list(zip(agg_fuzzy_names, agg_abc_element))
        for name, element in agg_fuzzy_set:
            self.A[name] = fuzz.trimf(self.A.universe, element)

        # 规则映射表（驾驶风格 -> [换道风险] -> [换道收益] -> 换道决策）
        rules_mapping = {
            '中性': [
                ['中', '较高', '高', '高', '高'],
                ['较低', '中', '较高', '高', '高'],
                ['低', '较低', '中', '较高', '高'],
                ['低', '低', '较低', '中', '较高'],
                ['低', '低', '低', '较低', '中']
            ],
            '激进': [
                ['较高', '高', '高', '高', '高'],
                ['中', '较高', '高', '高', '高'],
                ['较低', '中', '较高', '高', '高'],
                ['低', '较低', '中', '较高', '高'],
                ['低', '低', '较低', '中', '较高']
            ],
            '保守': [
                ['较低', '中', '较高', '高', '高'],
                ['低', '较低', '中', '较高', '高'],
                ['低', '低', '较低', '中', '较高'],
                ['低', '低', '低', '较低', '中'],
                ['低', '低', '低', '低', '较低']
            ]
        }

        # 生成规则
        rules = []
        for style, risk_levels in rules_mapping.items():
            # 打印驾驶风格，使用\t分隔每一个换道风险，使用join函数
            # for line in risk_levels:
            #     print(f"{'\t'.join(line)}")

            for i, risk in enumerate(fuzzy_names):  # 遍历换道风险 (低, 较低, 中, 较高, 高)
                for j, benefit in enumerate(fuzzy_names):  # 遍历换道收益
                    decision = rules_mapping[style][i][j]  # 获取对应的换道决策
                    rules.append(ctrl.Rule(self.A[style] & self.R[risk] & self.B[benefit], self.D[decision]))

        # 模糊控制系统
        lane_change_ctrl = ctrl.ControlSystem(rules)
        self.lane_change_sim = ctrl.ControlSystemSimulation(lane_change_ctrl)

    def compute(self, benefit, risk, aggressiveness):
        # 输入值
        self.lane_change_sim.input['收益'] = benefit
        self.lane_change_sim.input['风险'] = risk
        self.lane_change_sim.input['激进度'] = aggressiveness

        # 计算结果
        self.lane_change_sim.compute()
        return self.lane_change_sim.output['决策']

    def plot_membership_functions(self, variable):
        # 可视化变量隶属度函数
        # 设置默认matplotlib尺寸参数
        mm = 1 / 25.4  # mm转inch
        fontsize = 10  # 7磅/pt/point
        _width = 70 * mm  # 图片宽度英寸
        _ratio = 5 / 7  # 图片长宽比
        figsize = (_width, _width * _ratio)
        # rcParam
        plt.rcParams['figure.figsize'] = figsize

        self.B.view()  # 可视化变量隶属度函数
        fig = plt.gcf()
        fig.savefig("fuzzy_benefit.png", dpi=500, pil_kwargs={"compression": "tiff_lzw"})

        self.A.view()
        fig = plt.gcf()
        fig.savefig("fuzzy_aggressiveness.png", dpi=500, pil_kwargs={"compression": "tiff_lzw"})

        self.R.view()
        fig = plt.gcf()
        fig.savefig("fuzzy_risk.png", dpi=500, pil_kwargs={"compression": "tiff_lzw"})

        self.D.view()
        fig = plt.gcf()
        fig.savefig("fuzzy_decision.png", dpi=500, pil_kwargs={"compression": "tiff_lzw"})

        plt.show()


def test_fuzzy_logic():
    fuzzy_logic = FuzzyLogic()

    lane_change_sim = fuzzy_logic.lane_change_sim

    # 输入值
    lane_change_sim.input['收益'] = 1
    lane_change_sim.input['风险'] = 0
    lane_change_sim.input['激进度'] = 1

    # 计算结果
    lane_change_sim.compute()
    print("换道决策得分:", lane_change_sim.output['决策'])


    mm = 1 / 25.4  # mm转inch
    fontsize = 10  # 7磅/pt/point
    _width = 70 * 2 * mm  # 图片宽度英寸
    _ratio = 5 / 7  # 图片长宽比
    figsize = (_width, _width * _ratio)

    # 设置图形画布
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 定义换道风险和收益的范围
    risk_values = np.linspace(0, 1, 10)  # 换道风险
    benefit_values = np.linspace(0, 1, 10)  # 换道收益

    # 使用不同的激进度绘制图像
    for aggressiveness in [0.25, 0.5, 0.75]:
        # 存储每个激进度下的决策值
        decision_values = np.zeros((len(risk_values), len(benefit_values)))

        # 计算决策值
        for i, risk in enumerate(risk_values):
            for j, benefit in enumerate(benefit_values):
                lane_change_sim.input['激进度'] = aggressiveness
                lane_change_sim.input['风险'] = risk
                lane_change_sim.input['收益'] = benefit
                lane_change_sim.compute()
                decision_values[i, j] = lane_change_sim.output['决策']

        # 创建网格
        X, Y = np.meshgrid(risk_values, benefit_values)
        Z = decision_values

        # 绘制三维曲面
        ax.plot_surface(X, Y, Z, label=f'激进度: {aggressiveness}', alpha=0.5)

    # 设置图表标签
    ax.set_xlabel('换道收益')
    ax.set_ylabel('换道风险')
    ax.set_zlabel('换道决策得分')

    # 将换道风险轴反转
    ax.set_ylim(1, 0)

    # 显示图例
    ax.legend(loc='upper left', fontsize=fontsize)

    fig.savefig("fuzzy_decision_surface.png", dpi=500, pil_kwargs={"compression": "tiff_lzw"})

    # 显示图形
    plt.show()

