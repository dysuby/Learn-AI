from time import time
import numpy as np
from utils import read_tsp, genAnimation
from Solution import Solution


class LS:

    def __init__(self, tsppath):
        self.nodes = read_tsp(tsppath)

    def run(self, times, neighbor_size):
        """
        `times` - 迭代次数

        `neighbor_size` - 邻域大小
        """
        self.optimal = Solution.genInitSolution(self.nodes)

        self.s_set = [self.optimal]
        for i in range(times):
            new_solution = self.optimal.localsearch(neighbor_size)
            if new_solution.cost < self.optimal.cost:
                self.optimal = new_solution
                self.s_set.append(self.optimal)

            print('times {}/{}: current: {} best: {}'.format(
                i, times, new_solution.cost, self.optimal.cost))

        print('final path\n{}\ncost: {}'.format(
            self.optimal.path, self.optimal.cost))



if __name__ == '__main__':
    ls = LS('pr136.tsp')
    st = time()
    ls.run(200000, 1)
    et = time()
    print('Cost time: {} mins'.format((et - st) / 60))
    genAnimation(ls.s_set)