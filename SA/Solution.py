import numpy as np
import random


class Solution:
    @staticmethod
    def genInitSolution(distances, t='random'):
        # 生成初始解
        if t == 'random':
            s_index = np.arange(0, len(distances), dtype=np.int)
            np.random.shuffle(s_index)
        else:
            s_index = Solution.greedy(distances)

        # 计算初始解的代价
        cost = distances[s_index[:-1], s_index[1:]
                         ].sum() + distances[s_index[-1], s_index[0]]

        return Solution(distances, s_index, cost)

    @staticmethod
    def greedy(distances):
        start = random.randint(0, len(distances))
        path = [start]
        d = distances.copy()
        d[:, start] = np.inf
        while len(path) != len(distances):
            j = np.argmin(d[path[-1]])
            path.append(j)
            d[:, j] = np.inf
        return np.array(path, dtype=np.int)

    def __init__(self, distances, path, cost):
        self.path = path
        self.cost = cost
        self.dimension = len(self.path)
        self.distances = distances

    def localsearch(self):
        return random.sample([self.reverse, self.single_swap, self.swap], 1)[0]()

    def reverse(self):
        i, j = random.sample(range(0, self.dimension), 2)
        i, j = min([i, j]), max([i, j])

        # 0, ..., i-1, i, ..., j-1, j, ...
        # 逆序得
        # 0, ..., i-1, j-1, ..., i, j, ...
        new_path = self.path.copy()
        new_path[i:j] = new_path[i:j][::-1]

        new_cost = self.cost - self.distances[self.path[i], self.path[i - 1]]
        new_cost -= self.distances[self.path[j], self.path[j - 1]]
        new_cost += self.distances[self.path[j - 1], self.path[i - 1]]
        new_cost += self.distances[self.path[j], self.path[i]]

        return Solution(self.distances, new_path, new_cost)

    def swap(self):
        i, j, k = random.sample(range(0, self.dimension), 3)
        i, j, k = sorted([i, j, k])

        # 0, ..., i-1, i, ..., j-1, j, ..., k-1, k, ...
        # 交换得
        # 0, ..., i-1, j, ...k-1, i, ..., j-1, k, ...
        new_path = np.concatenate(
            [self.path[:i], self.path[j:k], self.path[i:j], self.path[k:]])

        new_cost = self.cost - self.distances[self.path[i], self.path[i - 1]]
        new_cost -= self.distances[self.path[j], self.path[j - 1]]
        new_cost -= self.distances[self.path[k], self.path[k - 1]]
        new_cost += self.distances[self.path[j], self.path[i - 1]]
        new_cost += self.distances[self.path[i], self.path[k - 1]]
        new_cost += self.distances[self.path[k], self.path[j - 1]]

        return Solution(self.distances, new_path, new_cost)

    def single_swap(self):
        i, j = random.sample(range(0, self.dimension), 2)
        i, j = min([i, j]), max([i, j])
        if j - i == self.dimension - 1:
            j, i = 0, -1

        # 0, ..., i, ..., j, ...
        # 交换得
        # 0, ..., j, ..., i, ...
        new_path = self.path.copy()
        new_path[i], new_path[j] = new_path[j], new_path[i]


        new_cost = self.cost
        if j - 1 != i:  # 不相邻
            new_cost -= self.distances[self.path[i],
                                    self.path[(i + 1) if i + 1 != self.dimension else 0]]
            new_cost -= self.distances[self.path[j], self.path[j - 1]]
            new_cost += self.distances[self.path[j],
                                    self.path[(i + 1) if i + 1 != self.dimension else 0]]
            new_cost += self.distances[self.path[i], self.path[j - 1]]

        new_cost -= self.distances[self.path[i], self.path[i - 1]]
        new_cost -= self.distances[self.path[j],
                                   self.path[(j + 1) if j + 1 != self.dimension else 0]]

        new_cost += self.distances[self.path[j], self.path[i - 1]]

        new_cost += self.distances[self.path[i],
                                   self.path[(j + 1) if j + 1 != self.dimension else 0]]

        return Solution(self.distances, new_path, new_cost)
