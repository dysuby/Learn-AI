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

        s_set = [self.optimal]
        for i in range(times):
            new_solution = self.optimal.localsearch(neighbor_size)
            if new_solution.cost < self.optimal.cost:
                self.optimal = new_solution
                s_set.append(self.optimal)

            print('times {}/{}: current: {} best: {}'.format(
                i, times, new_solution.cost, self.optimal.cost))

        print('final path\n{}\ncost: {}'.format(
            self.optimal.path, self.optimal.cost))

        genAnimation(s_set)


if __name__ == '__main__':
    LS('pr136.tsp').run(2000000, 10)
