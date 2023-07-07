from subprocess import Popen

from 仿真设置 import *


def 并行仿真() -> None:
    仿真脚本文件 = Path(__file__).parent / './逐个仿真.py'
    并行数量限制 = 4

    管道列表 = []
    for 变量 in 变量列表:
        for 变量值 in 变量值表[变量]:
            for 算法 in 算法列表:
                while True:
                    if len(管道列表) <= 并行数量限制:
                        break
                    else:
                        for 管道 in 管道列表:
                            if 管道.poll() is not None:
                                管道.kill()
                                管道列表.remove(管道)

                管道 = Popen(
                    ['python', str(仿真脚本文件), 变量, str(变量值), 算法]
                )
                管道列表.append(管道)

    for 管道 in 管道列表:
        管道.wait()


if __name__ == '__main__':
    并行仿真()
