from math import floor, ceil, sqrt
from typing import Tuple, Dict

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

from 流量生成 import 正常流量, 攻击流量
from 流量预测 import 正常流量预测, 攻击流量预测


class 模拟网络环境类(Env):

    def __init__(self, 最大负载: int = 10000,
                 正常流量倍数: float = 6, 攻击流量倍数: int = 1,
                 正常流量检测准确度: float = 0.9, 攻击流量检测准确度: float = 0.9,
                 调整周期: int = 1, 时间范围: slice = slice(100, 700),
                 是否记录: bool = False):
        super().__init__()

        self.action_space = Discrete(11)
        self.observation_space = Box(low=0, high=1e+5, shape=(2,), dtype=np.int32)

        self.最大负载 = 最大负载
        self.正常流量倍数 = 正常流量倍数
        self.攻击流量倍数 = 攻击流量倍数
        self.正常流量检测准确度 = 正常流量检测准确度
        self.攻击流量检测准确度 = 攻击流量检测准确度
        self.记录历史 = 是否记录

        self.原始正常流量 = 正常流量[时间范围]
        self.原始攻击流量 = 攻击流量[时间范围]
        self.原始正常流量预测 = 正常流量预测[时间范围]
        self.原始攻击流量预测 = 攻击流量预测[时间范围]

        self.正常流量 = np.round(self.正常流量倍数 * self.原始正常流量).astype(np.int32)
        self.攻击流量 = np.round(self.攻击流量倍数 * self.原始攻击流量).astype(np.int32)
        self.正常流量预测 = np.round(self.正常流量倍数 * self.原始正常流量预测).astype(np.int32)
        self.攻击流量预测 = np.round(self.攻击流量倍数 * self.原始攻击流量预测).astype(np.int32)
        self.设置调整周期(调整周期)
        self.总时长 = len(self.正常流量)

        self.观察值表 = {}
        self.正常流量通过表 = {}
        self.最佳丢弃比例表 = self.获取最佳丢弃比例表()
        self.奖励表 = {}

        self.步数 = 0

        self.动作历史 = []
        self.观察值历史 = []
        self.奖励历史 = []

    def step(self, 动作: float) -> Tuple[np.ndarray, float, bool, dict]:
        正常流量预测值 = self.正常流量预测[self.步数]
        攻击流量预测值 = self.攻击流量预测[self.步数]
        观察值 = np.array([正常流量预测值, 攻击流量预测值], dtype=np.int32)

        正常流量 = self.正常流量[self.步数]
        攻击流量 = self.攻击流量[self.步数]
        try:
            奖励 = self.奖励表[(正常流量, 攻击流量, 动作)]
        except KeyError:
            丢弃比例 = 动作 / (self.action_space.n - 1)
            通过的正常流量数量 = self.计算通过的正常流量数量(丢弃比例, 正常流量, 攻击流量)
            最佳丢弃比例 = self.最佳丢弃比例表[(正常流量, 攻击流量)]
            最多可能通过的正常流量数量 = self.计算通过的正常流量数量(最佳丢弃比例, 正常流量, 攻击流量)
            奖励 = 通过的正常流量数量 - 最多可能通过的正常流量数量
            self.奖励表[(正常流量, 攻击流量, 动作)] = 奖励

        是否结束 = True if self.步数 >= self.总时长 - 1 else False
        self.步数 += 1

        if self.记录历史:
            self.记录(动作, 观察值, 奖励)

        附加信息 = {}

        return 观察值, 奖励, 是否结束, 附加信息

    def reset(self, **kwargs) -> np.ndarray:
        self.步数 = 0
        return np.array([0, 0], dtype=np.int32)

    def render(self, mode="human") -> None:
        pass

    def 获取最佳丢弃比例表(self) -> Dict[Tuple[int, int], float]:
        最佳丢弃比例表 = {}

        for 正常流量数量 in set(self.正常流量):
            for 攻击流量数量 in set(self.攻击流量):
                最佳丢弃比例 = 0
                最多可能通过正常流量数量 = 0

                for 丢弃比例 in range(11):
                    丢弃比例 /= 10
                    通过正常流量数量 = self.计算通过的正常流量数量(丢弃比例, 正常流量数量, 攻击流量数量)
                    if 通过正常流量数量 > 最多可能通过正常流量数量:
                        最多可能通过正常流量数量 = 通过正常流量数量
                        最佳丢弃比例 = 丢弃比例
                最佳丢弃比例表[(正常流量数量, 攻击流量数量)] = 最佳丢弃比例

        return 最佳丢弃比例表

    def 计算通过的正常流量数量(self, 丢弃比例: float, 正常流量数量: int, 攻击流量数量: int) -> int:
        通过的正常流量数量 = 正常流量数量 * self.正常流量检测准确度 \
                             + 正常流量数量 * (1 - self.正常流量检测准确度) * (1 - 丢弃比例)
        通过的攻击流量数量 = 攻击流量数量 * self.攻击流量检测准确度 * (1 - 丢弃比例) \
                    + 攻击流量数量 * (1 - self.攻击流量检测准确度)
        总通过数量 = 通过的正常流量数量 + 通过的攻击流量数量

        if 总通过数量 > self.最大负载:
            通过的正常流量数量 = (self.最大负载 / 总通过数量) * 通过的正常流量数量

        通过的正常流量数量 = round(通过的正常流量数量)

        return 通过的正常流量数量

    def 记录(self, 动作: float, 观察值: np.ndarray, 奖励: float) -> None:
        self.动作历史.append(动作)
        self.观察值历史.append(观察值)
        self.奖励历史.append(奖励)

    def 清除记录(self) -> None:
        self.动作历史 = []
        self.观察值历史 = []
        self.奖励历史 = []

    def 设置调整周期(self, 周期: int) -> None:
        重新统计的正常流量 = []
        for i in range(floor(len(self.正常流量) / 周期)):
            重新统计的正常流量.append(self.正常流量[(i * 周期):((i + 1) * 周期)].sum())
        重新统计的正常流量 = np.array(重新统计的正常流量, np.int32)
        self.正常流量 = 重新统计的正常流量

        重新统计的攻击流量 = []
        for i in range(floor(len(self.攻击流量) / 周期)):
            重新统计的攻击流量.append(self.攻击流量[(i * 周期):((i + 1) * 周期)].sum())
        重新统计的攻击流量 = np.array(重新统计的攻击流量, np.int32)
        self.攻击流量 = 重新统计的攻击流量

        重新统计的正常流量预测 = []
        for i in range(floor(len(self.正常流量预测) / 周期)):
            重新统计的正常流量预测.append(self.正常流量预测[(i * 周期):((i + 1) * 周期)].sum())
        重新统计的正常流量预测 = np.array(重新统计的正常流量预测, np.int32)
        self.正常流量预测 = 重新统计的正常流量预测

        重新统计的攻击流量预测 = []
        for i in range(floor(len(self.攻击流量预测) / 周期)):
            重新统计的攻击流量预测.append(self.攻击流量预测[(i * 周期):((i + 1) * 周期)].sum())
        重新统计的攻击流量预测 = np.array(重新统计的攻击流量预测, np.int32)
        self.攻击流量预测 = 重新统计的攻击流量预测

        self.最大负载 = self.最大负载 * 周期


if __name__ == '__main__':
    网络环境 = 模拟网络环境类()
    网络环境.step(0)
    print('完成.')
