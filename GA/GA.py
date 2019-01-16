from time import time
import random
import numpy as np
from utils import read_tsp, genAnimation
from Solution import Solution

class GA:
    def __init__(self, tsppath):
        self.nodes = read_tsp(tsppath)
        # 计算距离
        self.distances = np.empty((len(self.nodes), len(self.nodes)))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                self.distances[i, j] = np.linalg.norm(self.nodes[i] - self.nodes[j])
            self.distances[i, i] = np.inf

    def tournament(self, gens, n):
        samples = random.sample(gens, n)
        return min(samples, key=lambda s: s.cost)

    def run(self, gens_num, pop_size, pc, pm, k):
        population = []
        greedy = Solution.genInitSolution(self.distances, 'greedy')
        for i in range(pop_size):
            if i > pop_size / 3:
                population.append(Solution.genInitSolution(self.distances, 'random'))
            else:
                population.append(greedy.localsearch())

        self.s_set = [min(population, key=lambda s: s.cost)]
        self.opt = population[0]

        for i in range(gens_num):
            # 产生新群体
            new_gen = []

            while len(new_gen) < pop_size:
                # 按概率交叉
                if np.random.random() < pc:
                    f1 = self.tournament(population, k)
                    f2 = self.tournament(population, k)
                    new_gen = new_gen + Solution.crossover(f1, f2)

                # 按概率变异
                if np.random.random() < pm:
                    f1 = self.tournament(population, k)
                    new_gen.append(min([f1, f1.localsearch()], key=lambda s: s.cost))

            population = new_gen

            # 记录当代最优解和总体最优解
            least = min(population, key=lambda s: s.cost)
            if least.cost < self.s_set[-1].cost:
                self.s_set.append(least)
                if self.opt.cost > least.cost:
                    self.opt = least
            print('G: {} Current opt cost: {} Best cost: {}'.format(i, least.cost, self.opt.cost))

        print('Final path:\n{}'.format(self.opt.path))
        print('Final cost: {}'.format(self.opt.cost))


def test(num):
    for i in range(num):
        ga = GA('ch130.tsp')
        st = time()
        ga.run(5000, 50, 0.9, 0.5, 5)
        et = time()
        print('Cost time: {} mins'.format((et - st) / 60))
        genAnimation(ga.s_set, ga.nodes)

if __name__ == '__main__':
    test(1)
