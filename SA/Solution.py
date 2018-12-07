import numpy as np


class Solution:
    @staticmethod
    def genInitSolution(nodes):
        # 生成初始解：[(x1, y1), (x2, y2), ..., (xn, yn)]
        s_index = np.arange(0, len(nodes), dtype=np.int)
        np.random.shuffle(s_index)
        path = nodes[s_index]
        # 计算初始解的代价
        cost = 0
        for i in range(1, len(path)):
            cost += np.linalg.norm(path[i] -
                                   path[i - 1])
        cost += np.linalg.norm(path[0] -
                               path[-1])

        return Solution(path, cost)

    def __init__(self, path, cost):
        self.path = path
        self.cost = cost
        self.dimension = len(self.path)

    def localsearch(self, neighbor_size, depth=1):
        neighbors = self.getNeighbors(neighbor_size, depth)
        opt = min(neighbors, key=lambda s: s.cost)
        return opt

    def getNeighbors(self, neighbor_size, depth):
        if depth == 0:
            return []

        ret = []
        for k in range(neighbor_size):
            ret = ret + [self.reverse(), self.swap()]
            ret = ret + ret[-1].getNeighbors(neighbor_size, depth - 1) + \
                ret[-2].getNeighbors(neighbor_size, depth - 1)

        return ret

    def reverse(self):
        i, j = np.random.randint(0, self.dimension, 2)
        while i == j:
            i, j = np.random.randint(0, self.dimension, 2)
        i, j = min([i, j]), max([i, j])

        # 0, ..., i-1, i, ..., j-1, j, ...
        # 逆序得
        # 0, ..., i-1, j-1, ..., i, j, ...
        new_path = np.concatenate(
            [self.path[:i], self.path[i:j][::-1], self.path[j:]])

        new_cost = self.cost - \
            np.linalg.norm(self.path[i] - self.path[i - 1])
        new_cost -= np.linalg.norm(self.path[j] - self.path[j - 1])
        new_cost += np.linalg.norm(self.path[j - 1] - self.path[i - 1])
        new_cost += np.linalg.norm(self.path[j] - self.path[i])

        return Solution(new_path, new_cost)

    def swap(self):
        i, j, k = np.random.randint(0, self.dimension, 3)
        while i == j or j == k or i == k:
            i, j, k = np.random.randint(0, self.dimension, 3)
        i, j, k = sorted([i, j, k])

        # 0, ..., i-1, i, ..., j-1, j, ..., k-1, k, ...
        # 交换得
        # 0, ..., i-1, j, ...k-1, i, ..., j-1, k, ...
        new_path = np.concatenate(
            [self.path[:i], self.path[j:k], self.path[i:j], self.path[k:]])

        new_cost = self.cost - \
            np.linalg.norm(self.path[i] - self.path[i - 1])
        new_cost -= np.linalg.norm(self.path[j] - self.path[j - 1])
        new_cost -= np.linalg.norm(self.path[k] - self.path[k - 1])
        new_cost += np.linalg.norm(self.path[j] - self.path[i - 1])
        new_cost += np.linalg.norm(self.path[i] - self.path[k - 1])
        new_cost += np.linalg.norm(self.path[k] - self.path[j - 1])

        return Solution(new_path, new_cost)
