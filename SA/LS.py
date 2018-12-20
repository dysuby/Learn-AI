from time import time
import numpy as np
from utils import read_tsp, genAnimation
from Solution import Solution


class LS:

    def __init__(self, tsppath):
        self.nodes = read_tsp(tsppath)
        # 计算距离
        self.distances = np.empty((len(self.nodes), len(self.nodes)))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                self.distances[i, j] = np.linalg.norm(
                    self.nodes[i] - self.nodes[j])
            self.distances[i, i] = np.inf

    def run(self, times):
        """
        `times` - 迭代次数

        `neighbor_size` - 邻域大小
        """
        self.optimal = Solution.genInitSolution(self.distances)

        self.s_set = [self.optimal]
        for i in range(times):
            new_solution = self.optimal.localsearch()
            if new_solution.cost < self.optimal.cost:
                self.optimal = new_solution
                self.s_set.append(self.optimal)

            print('times {}/{}: current: {} best: {}'.format(
                i, times, new_solution.cost, self.optimal.cost))

        print('final path\n{}\ncost: {}'.format(
            self.optimal.path, self.optimal.cost))

        cost = 0
        for i in range(-1, len(self.optimal.path) - 1):
            cost += np.sqrt(np.square(self.nodes[self.optimal.path[i]] - \
                self.nodes[self.optimal.path[i + 1]]).sum())
        print('Cost {}'.format(cost))


if __name__ == '__main__':
    ls = LS('ch130.tsp')
    st = time()
    ls.run(200000)
    et = time()
    print('Cost time: {} mins'.format((et - st) / 60))
    genAnimation(ls.s_set, ls.nodes)
