import sys

from 仿真 import 仿真
from 仿真设置 import *

if __name__ == '__main__':
    变量 = sys.argv[1]
    变量值 = eval(sys.argv[2])
    算法 = sys.argv[3]
    仿真配置列表 = [仿真配置表[算法][变量][变量值]]
    仿真(仿真配置列表)
    print(f'{变量}{变量值}{算法}仿真完成.')
