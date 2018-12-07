import numpy as np
from utils import read_tsp, genAnimation
from Solution import Solution


class SA:

    def __init__(self, tsppath):
        self.nodes = read_tsp(tsppath)

    def run(self, tmin, ntimes, neighbor_size, T, T_ratio, n_rate):
        """
        `tmin` - 最低温

        `ntimes` - 当前温度迭代次数
        
        `neighbor_size` - 邻域大小
        
        `T` - 初始温度
        
        `T_ratio` - 降温速率 (`T *= ratio`)
        
        `n_rate` - 降温时迭代次数改变速率 (`ntimes *= n_rate`)
        """
        solution = Solution.genInitSolution(self.nodes)
        self.optimal = solution

        s_set = [solution]
        while T > tmin:
            for j in range(ntimes):
                new_solution = solution.localsearch(neighbor_size)
                delta = new_solution.cost - solution.cost

                if delta <= 0 or np.random.ranf() < np.exp(-delta / T):
                    solution = new_solution
                    if solution.cost < self.optimal.cost:
                        self.optimal = solution
                        s_set.append(solution)

                print('T: {} times: {}/{}: current: {} best: {}'.format(
                    T, j, ntimes, solution.cost, self.optimal.cost))

            T *= T_ratio
            ntimes = int(n_rate * ntimes)

        print('final path\n{}\ncost: {}'.format(
            self.optimal.path, self.optimal.cost))

        genAnimation(s_set)


if __name__ == '__main__':
    SA('pr136.tsp').run(10, 1000, 150, 300, 0.98, 1.01)
