from typing import Dict

import joblib

from 仿真设置 import *
from 模拟网络环境 import 模拟网络环境类

基本正常流量总数 = 模拟网络环境类(时间范围=slice(700, 1300)).正常流量.sum()


def 统计总损失(测试序号: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    总损失 = {}
    for 变量 in 变量列表:
        总损失[变量] = {}
        for 变量值 in 变量值表[变量]:
            总损失[变量][变量值] = {}
            for 算法 in 算法列表:
                测试记录 = joblib.load(
                    Path(__file__).parent.parent / f'数据/测试{测试序号}/测试记录/{变量}{变量值}{算法}'
                )

                if 变量 != '正常流量倍数':
                    正常流量总数 = 基本正常流量总数
                else:
                    正常流量总数 = 模拟网络环境类(**{变量: 变量值}, 时间范围=slice(700, 1300)).正常流量.sum()

                总损失[变量][变量值][算法] = - 测试记录[2].sum() / 正常流量总数
    return 总损失


def 对比方法的极端值(总损失: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    极端值总损失 = {}
    for 变量 in 变量列表:
        极端值总损失[变量] = {}
        for 变量值 in (变量值表[变量][0], 变量值表[变量][-1]):
            极端值总损失[变量][变量值] = {}
            for 算法 in 算法列表:
                极端值总损失[变量][变量值][算法] = 总损失[变量][变量值][算法]
    return 极端值总损失


if __name__ == '__main__':
    测试序号 = 5
    总损失 = 统计总损失(测试序号)
    极端值总损失 = 对比方法的极端值(总损失)
    print(极端值总损失)
