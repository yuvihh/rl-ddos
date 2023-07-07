from abc import ABC, abstractmethod
from typing import Tuple, Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from 模拟网络环境 import 模拟网络环境类


class 模型基类(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def learn(self, *args, **kwargs) -> None:
        pass

    def save(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, 观察值: np.ndarray) -> Tuple[int, Any]:
        pass


class 全丢模型(模型基类):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, 观察值: np.ndarray) -> Tuple[int, Any]:
        return 10, None


class 预测最佳丢弃比例模型(模型基类):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.最大负载 = kwargs['env'].envs[0].最大负载
        self.正常流量检测准确度 = kwargs['env'].envs[0].正常流量检测准确度
        self.攻击流量检测准确度 = kwargs['env'].envs[0].攻击流量检测准确度

    def predict(self, 观察值: np.ndarray) -> Tuple[float, Any]:
        正常流量 = 观察值[0]
        攻击流量 = 观察值[1]

        if 正常流量 + 攻击流量 < self.最大负载:
            最佳丢弃率 = 0
        elif self.正常流量检测准确度 * 正常流量 + (1 - self.攻击流量检测准确度) * 攻击流量 <= self.最大负载:
            最佳丢弃率 = (正常流量 + 攻击流量 - self.最大负载) \
                    / ((1 - self.正常流量检测准确度) * 正常流量 + self.攻击流量检测准确度 * 攻击流量)
        else:
            最佳丢弃率 = 1

        动作 = 最佳丢弃率 * 10

        return 动作, None


class 决策树回归模型(模型基类):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.环境: 模拟网络环境类 = kwargs['env'].envs[0]
        self.流水线 = make_pipeline(StandardScaler(), DecisionTreeRegressor())

    def learn(self, *args, **kwargs) -> None:
        特征矩阵 = np.array([
            self.环境.正常流量,
            self.环境.攻击流量
        ]).T
        目标向量 = np.zeros_like(self.环境.正常流量, dtype=np.float64)
        for 时间 in range(self.环境.总时长):
            目标向量[时间] = self.环境.最佳丢弃比例表[(self.环境.正常流量[时间], self.环境.攻击流量[时间])]
        self.流水线.fit(特征矩阵, 目标向量)

    def predict(self, 观察值: np.ndarray) -> Tuple[int, Any]:
        return round(10 * self.流水线.predict(np.array([观察值]))[0]), None
