import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def read_tsp(tsppath):
    """
    解析 tsp，返回节点坐标

    @return - np.ndarray
    """
    SIZE = 1024
    f = open(tsppath)
    for i in range(6):
        f.readline(SIZE)

    ret = np.empty((0, 2), dtype=np.int32)
    for data in f:
        if data.find('EOF') != -1:
            break
        x, y = [int(s) for s in data.split()][1:]
        ret = np.append(ret, [[x, y]], axis=0)

    return ret


def genAnimation(solutions):
    """
    生成动画并保存 gif

    `solutions` - `Solution` 数组
    """
    fig, ax = plt.subplots()

    container = []
    for s in solutions:
        # 首尾相连
        X = np.append([s.path[-1, 0]], s.path[:, 0])
        Y = np.append([s.path[-1, 1]], s.path[:, 1])

        current, = ax.plot(X, Y, '-bo')

        # 标出 cost
        title = plt.text(0.5, 1.05, 'Cost {}'.format(s.cost),
                         size=plt.rcParams['axes.titlesize'],
                         ha='center', va='top', transform=ax.transAxes)
        container.append([current, title])

    ani = animation.ArtistAnimation(fig, container, blit=False, repeat=False)

    plt.show()
    print('saving to output.gif...')
    ani.save('output.gif', writer='pillow')
