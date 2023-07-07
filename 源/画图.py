from typing import List
import os

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from 仿真设置 import *
from 模拟网络环境 import 模拟网络环境类


算法标记 = {
    'PPO+ARIMA': 'o',
    'ARIMA': '^',
    '全丢': '*',
    'ARIMA+决策树回归': 'v'
}

算法图例 = {
    'PPO+ARIMA': 'our method',
    'ARIMA': 'method 2',
    '全丢': 'all drop',
    'ARIMA+决策树回归': 'ARIMA+DT'
}

变量x轴标签 = {
    '攻击流量倍数': 'attack traffic multiple',
    '正常流量倍数': 'benign traffic multiple',
    '正常流量检测准确度': 'benign detection accuracy',
    '攻击流量检测准确度': 'ddos detection accuracy',
    '调整周期': 'interval'
}

算法列表 = [
    '全丢',
    'ARIMA',
    'PPO+ARIMA',
    # 'ARIMA+决策树回归'
]

单位时长 = 20

基本正常流量总数 = 模拟网络环境类(时间范围=slice(700, 1300)).正常流量.sum()


def 算法性能随变量改变画图(测试序号: int) -> None:
    for 变量 in 变量列表:
        图, 坐标轴 = plt.subplots()

        for 算法 in 算法列表:
            总损失列表 = []
            for 变量值 in 变量值表[变量]:

                if 变量 == '正常流量倍数':
                    正常流量总数 = 模拟网络环境类(**{变量: 变量值}, 时间范围=slice(700, 1300)).正常流量.sum()
                else:
                    正常流量总数 = 基本正常流量总数

                try:
                    测试记录 = joblib.load(Path(__file__).parent.parent / f'数据/测试{测试序号}/测试记录/{变量}{变量值}{算法}')
                    总损失 = - 测试记录[2].sum() / 正常流量总数
                    总损失列表.append(总损失)
                except FileNotFoundError:
                    总损失列表.append(0)
                    print(f'{变量}{变量值}{算法}测试记录未找到.')

            坐标轴.plot(
                变量值表[变量],
                总损失列表,
                label=算法图例[算法],
                linewidth=1,
                marker=算法标记[算法]
            )

        坐标轴.set_xlabel(变量x轴标签[变量])
        坐标轴.set_ylabel('benign traffic loss')

        坐标轴.legend()

        图.savefig(Path(__file__).parent.parent / f'数据/测试{测试序号}/图/{变量}.png', bbox_inches='tight')
        print(f'{变量}画图完成.')


def 基本对比图(测试序号: int) -> None:
    # 图, 坐标轴 = plt.subplots(3, sharex='all', figsize=(9.6, 7.2))
    图, 坐标轴 = plt.subplots(4, sharex='all', figsize=(6.4, 6.4))
    坐标轴: plt.Axes
    图.subplots_adjust(right=0.7)

    变量 = '正常流量倍数'
    变量值 = 6

    模拟网络环境 = 模拟网络环境类(**{变量: 变量值})

    坐标轴[0].plot(np.arange(模拟网络环境.总时长) * 单位时长, 模拟网络环境.正常流量, label='BENIGN')
    坐标轴[0].plot(np.arange(模拟网络环境.总时长) * 单位时长, 模拟网络环境.攻击流量, label='DDoS')
    坐标轴[0].set_ylabel('traffic')
    坐标轴[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    坐标轴[1].plot(np.arange(模拟网络环境.总时长) * 单位时长, 模拟网络环境.正常流量预测, label='BENIGN')
    坐标轴[1].plot(np.arange(模拟网络环境.总时长) * 单位时长, 模拟网络环境.攻击流量预测, label='DDoS')
    坐标轴[1].set_ylabel('predicted traffic')

    for 算法 in 算法列表:
        测试记录 = joblib.load(Path(__file__).parent.parent / f'数据/测试{测试序号}/测试记录/{变量}{变量值}{算法}')
        动作记录 = 测试记录[0]
        流量损失 = - 测试记录[2]
        坐标轴[2].plot(np.arange(模拟网络环境.总时长) * 单位时长, 动作记录 / 10, label=算法图例[算法])
        坐标轴[2].set_ylabel('drop ratio')
        坐标轴[3].plot(np.arange(模拟网络环境.总时长) * 单位时长, 流量损失, label=算法图例[算法], linewidth=1)
        坐标轴[3].set_ylabel('benign loss')

    坐标轴[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    坐标轴[3].set_xlabel('time/s')

    # 坐标轴[0].set_xlim(0, 模拟网络环境.总时长 * 单位时长)
    # 坐标轴[0].set_ylim(0, 1.1e+4)
    # 坐标轴[1].set_ylim(0, 1.1)
    坐标轴[1].set_ylim(ymin=坐标轴[0].viewLim.ymin, ymax=坐标轴[0].viewLim.ymax)
    坐标轴[3].set_ylim(-50, 2000)

    坐标轴[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    坐标轴[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    坐标轴[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # 坐标轴[2].set_yscale('log', base=2)

    图.savefig(Path(__file__).parent.parent / f'数据/测试{测试序号}/图/基本对比图.png', bbox_inches='tight')
    print('基本对比图完成.')


def 画对比条形图(测试序号: int) -> None:
    for 变量 in 变量列表:
        图, 坐标轴 = plt.subplots()
        坐标轴: plt.Axes

        # 从变量值表里均匀取一半的值, 以减少点数量.
        变量值表[变量] = 变量值表[变量][::2]

        x轴位置 = np.arange(len(变量值表[变量]))
        宽度 = 0.2

        for 算法序号 in range(len(算法列表)):
            算法 = 算法列表[算法序号]
            总损失列表 = []
            for 变量值 in 变量值表[变量]:

                if 变量 == '正常流量倍数':
                    正常流量总数 = 模拟网络环境类(**{变量: 变量值}, 时间范围=slice(700, 1300)).正常流量.sum()
                else:
                    正常流量总数 = 基本正常流量总数

                try:
                    测试记录 = joblib.load(Path(__file__).parent.parent / f'数据/测试{测试序号}/测试记录/{变量}{变量值}{算法}')
                    总损失 = - 测试记录[2].sum() / 正常流量总数
                    总损失列表.append(总损失)
                except FileNotFoundError:
                    总损失列表.append(0)
                    print(f'{变量}{变量值}{算法}测试记录未找到.')

            坐标轴.bar(
                x轴位置 - 宽度 + 宽度 * 算法序号,
                总损失列表,
                宽度,
                label=算法图例[算法],
                edgecolor='black',
                linewidth=0.5
            )

        坐标轴.set_xlabel(变量x轴标签[变量])
        坐标轴.set_ylabel('benign traffic loss')
        坐标轴.set_xticks(x轴位置, 变量值表[变量])
        # 坐标轴.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        坐标轴.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        坐标轴.legend()

        图.savefig(Path(__file__).parent.parent / f'数据/测试{测试序号}/图/{变量}-条形图.png', bbox_inches='tight')
        print(f'{变量}条形图画图完成.')


def 画正常流量损失直方图(测试序号: int) -> None:
    图, 坐标轴 = plt.subplots(3, sharex='all', sharey='all')
    图: plt.Figure
    坐标轴: List[plt.Axes]

    变量 = '正常流量倍数'
    变量值 = 6
    算法 = 'ARIMA'
    测试记录 = joblib.load(Path(__file__).parent.parent / f'数据/测试{测试序号}/测试记录/{变量}{变量值}{算法}')
    # 分箱 = np.histogram_bin_edges(- 测试记录[2], bins='auto')
    分箱 = np.arange(0, 1000 + 25, 25)

    for 算法序号 in range(len(算法列表)):
        算法 = 算法列表[算法序号]
        测试记录 = joblib.load(Path(__file__).parent.parent / f'数据/测试{测试序号}/测试记录/{变量}{变量值}{算法}')
        流量损失 = - 测试记录[2]

        坐标轴[算法序号].hist(流量损失, 分箱, density=True)
        坐标轴[算法序号].set_ylim(5e-5, 1e-1)
        坐标轴[算法序号].set_yscale('log')

    坐标轴[算法序号].set_xlabel('benign traffic loss')
    坐标轴[1].set_ylabel('percentage')
    坐标轴[0].set_title('benign traffic loss histogram')

    图.savefig(Path(__file__).parent.parent / f'数据/测试{测试序号}/图/正常流量损失直方图.png', bbox_inches='tight')
    print('正常流量损失直方图完成.')


def 画ARIMA时序预测对比图(测试序号: int) -> None:
    模拟网络环境 = 模拟网络环境类(正常流量倍数=3)
    正常流量 = 模拟网络环境.正常流量
    正常流量预测 = 模拟网络环境.正常流量预测

    图, 坐标轴 = plt.subplots()

    坐标轴.plot(np.arange(模拟网络环境.总时长) * 单位时长, 正常流量, label='real traffic', linewidth=1)
    坐标轴.plot(np.arange(模拟网络环境.总时长) * 单位时长, 正常流量预测, label='prediction', linewidth=1)

    坐标轴.set_xlabel('time/s')
    坐标轴.set_ylabel('traffic')
    坐标轴.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    坐标轴.legend()

    图.savefig(Path(__file__).parent.parent / f'数据/测试{测试序号}/图/ARIMA时序预测图.png', bbox_inches='tight')
    print('ARIMA时序预测图完成.')


if __name__ == '__main__':
    测试序号 = 6
    算法性能随变量改变画图(测试序号)
    基本对比图(测试序号)
    画对比条形图(测试序号)
    画正常流量损失直方图(测试序号)
    画ARIMA时序预测对比图(测试序号)
