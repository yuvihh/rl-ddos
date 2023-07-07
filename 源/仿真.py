from typing import Tuple

import joblib
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import DummyVecEnv

from 模拟网络环境 import 模拟网络环境类
from 对比方法 import 模型基类


def 测试模型(模型: BaseRLModel or 模型基类, 环境: 模拟网络环境类) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在环境中使用模型, 返回动作, 观察值, 奖励列表."""
    总步数 = 环境.总时长
    动作数组 = np.zeros((总步数,))
    观察数组 = np.zeros((总步数, 环境.observation_space.shape[0]), dtype=np.int32)
    奖励数组 = np.zeros((总步数,))

    观察值 = 环境.reset()
    for 步数 in range(总步数):
        动作, 状态 = 模型.predict(观察值)
        观察值, 奖励, 是否结束, 附加信息 = 环境.step(int(动作))

        动作数组[步数] = 动作
        观察数组[步数] = 观察值
        奖励数组[步数] = 奖励

    return 动作数组, 观察数组, 奖励数组


def 仿真(仿真配置列表):
    """训练并测试模型, 返回测试的总奖励."""
    for 仿真配置 in 仿真配置列表:
        if 仿真配置['模型设置']['模型'] == PPO2:
            训练环境 = DummyVecEnv([模拟网络环境类 for 并行环境 in range(仿真配置['并行环境数'])])
        else:
            训练环境 = DummyVecEnv([模拟网络环境类])
        for 环境序号 in range(训练环境.num_envs):
            训练环境.envs[环境序号] = 模拟网络环境类(**仿真配置['环境参数'])
        模型: PPO2 = 仿真配置['模型设置']['模型'](env=训练环境, **仿真配置['模型设置']['模型参数'])
        模型.learn(**仿真配置['模型设置']['学习参数'])
        模型.save(仿真配置['保存设置']['模型保存位置'])

        测试环境 = 模拟网络环境类(**仿真配置['环境参数'], 时间范围=slice(700, 1300))
        测试记录 = 测试模型(模型, 测试环境)
        joblib.dump(测试记录, 仿真配置['保存设置']['测试记录保存位置'])
