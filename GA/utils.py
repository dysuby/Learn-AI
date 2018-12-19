import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TSP_PATH = 'test'


def read_tsp(tsppath):
    """
    解析 tsp，返回节点坐标

    @return - np.ndarray
    """
    SIZE = 1024
    f = open('{}/{}'.format(TSP_PATH, tsppath))
    while True:
        s = f.readline(SIZE)
        if s.find('NODE_COORD_SECTION') != -1:
            break

    ret = []
    for data in f:
        if data.find('EOF') != -1:
            break
        x, y = [float(s) for s in data.split()][1:]
        ret.append([x, y])

    return np.array(ret)


def genAnimation(solutions, nodes):
    """
    生成动画并保存 gif

    `solutions` - `Solution` 数组
    """
    fig, ax = plt.subplots()

    container = []
    i = 0
    for s in solutions:
        # 首尾相连
        X = np.append([nodes[s.path[-1], 0]], nodes[s.path, 0])
        Y = np.append([nodes[s.path[-1], 1]], nodes[s.path, 1])

        current, = ax.plot(X, Y, '-bo')

        # 标出 cost
        title = plt.text(0.5, 1.05, 'times: {} Cost {}'.format(i, s.cost),
                         size=plt.rcParams['axes.titlesize'],
                         ha='center', va='top', transform=ax.transAxes)
        container.append([current, title])
        i += 1

    ani = animation.ArtistAnimation(fig, container, blit=False, repeat=False)

    plt.show()
    print('saving to output.gif...')
    ani.save('output.gif', writer='pillow')
