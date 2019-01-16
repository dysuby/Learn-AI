import random

class AStar:
    def __init__(self, puzzle_type=8):
        self.end_state = list(range(9))
        self.puzzle_type = puzzle_type

    def is_valid(self, state):
        rev = 0
        for i in range(9):
            for j in range(i + 1, 9):
                rev += int(state[i] > state[j] and state[j] != 0)
        return not (rev & 1)

    def neighbors(self, state):
        z_idx = state.index(0)
        action = []
        if not z_idx < 3:
            action.append([-1, 0])
        if not z_idx >= 6:
            action.append([1, 0])
        if not z_idx % 3 == 0:
            action.append([0, -1])
        if not z_idx % 3 == 2:
            action.append([0, 1])
        ret = []
        for act in action:
            nei = state.copy()
            nei[z_idx], nei[z_idx + act[0] * 3 + act[1]
                            ] = nei[z_idx + act[0] * 3 + act[1]], nei[z_idx]
            ret.append(nei)
        return ret

    def h1(self, state):
        """
        错位数码数
        """
        return len([num for num in range(9) if state[num] != num and 
            not (self.puzzle_type == 8 and state[num] == 0)])

    def h2(self, state):
        """
        曼哈顿距离之和
        """
        err = 0
        for i in range(9):
            if not (self.puzzle_type == 8 and state[i] == 0):
                err += abs(state[i] - i) // 3 + abs(state[i] - i) % 3
        return err

    def search(self, init_state, h_type=1):
        if not self.is_valid(init_state):
            print('No solution for this init state')
            exit(1)
        # 选择 h
        h = [self.h1, self.h2][h_type - 1]
        
        dist = .1

        # 初始化表
        open_table = [init_state]
        closed = []
        G = [init_state]            # 记录出现过的结点
        prev = [-1]                 # 前驱结点
        g_score = [0]
        h_score = [h(open_table[0])]
        f = [h_score[0]]            # 估价结果
        
        while len(open_table):  # 当 Open 不为空
            # 寻找 Open 中最优结点
            min_state = min(open_table, key=lambda s: f[G.index(s)])
            idx = G.index(min_state)

            print('Open: {}'.format(len(open_table)))
            print('total node {}'.format(len(G)))
            print('min node:')
            self.print_state(min_state)

            # if f[idx] > f[0]:
            #     print('Not satisfy f(n)<=f*(n)')
            #     exit(1)

            # 计算前驱结点
            tree = []
            father = prev[idx]
            while father != -1:
                tree.append(G[father])
                father = prev[father]
            
            # 如果已达到结束状态
            if min_state == self.end_state:
                return open_table, list(reversed(tree)) + [min_state], G, f
            
            # 加入 Closed 并从 Open 中移除
            closed.append(min_state)
            open_table.remove(min_state)

            # 计算后继结点中不是前驱结点的 M
            neighbors = [n for n in self.neighbors(min_state) if n not in tree]
            for n in neighbors:
                if n not in G:
                    # 将 P 加入 Open 并计入 G 和 Tree
                    son_idx = -1
                    open_table.append(n)
                    G.append(n)
                    prev.append(idx)
                    g_score.append(g_score[idx] + dist)
                    h_score.append(h(n))
                    f.append(g_score[-1] + h_score[-1])
                else:
                    # 更改指针
                    son_idx = G.index(n)
                    if g_score[son_idx] > g_score[idx] + dist:
                        # 更优
                        prev[son_idx] = idx
                        g_score[son_idx] = g_score[idx] + dist
                        f[son_idx] = g_score[son_idx] + h_score[son_idx]
                # if self.puzzle_type == 8 and h_type == 1 and h_score[idx] > h_score[son_idx] + 1:
                #     print('Not satisfy h(ni）<= 1 + h(nj)')
                #     exit(1)

    def print_state(self, state):
        to_print = lambda num: num if num != 0 else ' '
        for i in range(0, 9, 3):
            print(to_print(state[i]), to_print(state[i+1]), to_print(state[i+2]))
        print('')

def gen_random_state():
    state = list(range(9))
    random.shuffle(state)
    return state

if __name__ == '__main__':
    a = AStar(puzzle_type=9)
    
    # 打乱
    state = list(range(9))
    for i in range(100):
        state, = random.sample(a.neighbors(state), 1)
    
    # 搜索
    open_table, path, G, f = a.search(state, h_type=1)
    
    print('-------------- Open table --------------')
    for o in open_table:
        a.print_state(o)
    print('-------------- path --------------')
    for p in path:
        print('f(n): {}'.format(f[G.index(p)]))
        a.print_state(p)
    
    # print('-------------- node in open_table and path --------------')
    # for o in open_table:
    #     if o in path:
    #         a.print_state(o)
    #         print('f(n): {}'.format(f[G.index(o)]))
