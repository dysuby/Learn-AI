from time import time
import numpy as np
from utils import read_tsp, genAnimation
from Solution import Solution


class SA:

    def __init__(self, tsppath):
        self.nodes = read_tsp(tsppath)
        self.s_set = []
        # 计算距离
        self.distances = np.empty((len(self.nodes), len(self.nodes)))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                self.distances[i, j] = np.linalg.norm(
                    self.nodes[i] - self.nodes[j])
            self.distances[i, i] = np.inf

    def run(self, tmin, ntimes, T, T_ratio, n_rate):
        """
        `tmin` - 最低温

        `ntimes` - 当前温度迭代次数
        
        `T` - 初始温度
        
        `T_ratio` - 降温速率 (`T *= ratio`)
        
        `n_rate` - 降温时迭代次数改变速率 (`ntimes *= n_rate`)
        """
        t = T
        solution = Solution.genInitSolution(self.distances)
        self.optimal = solution

        self.s_set = [solution]
        self.current_cost = []
        self.opt_cost = []
        while t > tmin:
            for j in range(ntimes):
                new_solution = solution.localsearch()
                delta = new_solution.cost - solution.cost
                if delta <= 0 or np.random.random() < np.exp(-delta / t):
                    solution = new_solution
                    if solution.cost < self.optimal.cost:
                        self.optimal = solution
                        self.s_set.append(solution)

                print('T: {} times: {}/{}: current: {} best: {}'.format(
                    t, j, ntimes, solution.cost, self.optimal.cost))

            t *= T_ratio
            ntimes = int(n_rate * ntimes)

        print('final path\n{}\ncost: {}'.format(
            self.optimal.path, self.optimal.cost))



if __name__ == '__main__':
    sa = SA('ch130.tsp')
    st = time()
    sa.run(1, 3000, 100, 0.97, 1)
    et = time()
    print('Cost time: {} mins'.format((et - st) / 60))
    genAnimation(sa.s_set, sa.nodes)
