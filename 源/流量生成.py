from math import ceil
from pathlib import Path

import joblib
import numpy as np


def 生成流量() -> (np.ndarray, np.ndarray):
    正常流量 = np.array(
        joblib.load(
            Path(__file__).parent.parent / '数据/流量数据/正常流量20s.joblib'
        )
    )
    攻击流量 = np.array(
        [
            *[0 for i in range(10)],
            *[1000 * i for i in range(10)],
            *[10000 for i in range(20)],
            *[10000 - 1000 * i for i in range(10)]
        ]
    )
    攻击流量 = np.tile(攻击流量, ceil(len(正常流量) / len(攻击流量)))[:len(正常流量)]
    return 正常流量, 攻击流量


正常流量, 攻击流量 = 生成流量()
