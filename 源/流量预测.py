from pathlib import Path

import joblib
import numpy as np
# from sktime.forecasting.arima import AutoARIMA

from 流量生成 import 正常流量, 攻击流量


def 预测正常流量() -> np.ndarray:
    开始时间 = 100
    窗口长度 = 100

    预测的正常流量 = np.zeros(正常流量.shape, dtype=np.int32)

    for 时间 in range(开始时间, len(正常流量)):
        预测模型 = AutoARIMA(suppress_warnings=True)
        预测模型.fit(正常流量[时间 - 窗口长度:时间])
        预测的正常流量[时间] = round(预测模型.predict(1)[0, 0])

        print(f'时间: {时间}, 真实流量: {正常流量[时间]}, 预测流量: {预测的正常流量[时间]}')

    return 预测的正常流量


def 预测攻击流量() -> np.ndarray:
    开始时间 = 100
    窗口长度 = 100

    预测的攻击流量 = np.zeros(攻击流量.shape, dtype=np.int32)

    for 时间 in range(开始时间, len(攻击流量)):
        预测模型 = AutoARIMA(suppress_warnings=True)
        预测模型.fit(攻击流量[时间 - 窗口长度:时间])
        预测的攻击流量[时间] = round(预测模型.predict(1)[0, 0])

        print(f'时间: {时间}, 真实流量: {攻击流量[时间]}, 预测流量: {预测的攻击流量[时间]}')

    return 预测的攻击流量


正常流量预测 = joblib.load(Path(__file__).parent.parent / '数据/流量数据/预测正常流量20s.joblib')
攻击流量预测 = joblib.load(Path(__file__).parent.parent / '数据/流量数据/预测攻击流量.joblib')


if __name__ == '__main__':
    # 预测的正常流量 = 预测正常流量()
    预测的攻击流量 = 预测攻击流量()

    # joblib.dump(预测的正常流量, Path(__file__).parent.parent / '数据/流量数据/预测正常流量20s.joblib')
    joblib.dump(预测的攻击流量, Path(__file__).parent.parent / '数据/流量数据/预测攻击流量.joblib')

    print('完成.')
