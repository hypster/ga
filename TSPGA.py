import os
from random import random
from random import randint
from GA import GA
from comp_func import softmax
from util import read_problem_tsp, read_tsp_solution
import numpy as np
from data.tsp import or_example


def dist(v, u):
    '''
    calculate the eucledian distance between two nodes
    :param v:
    :param u:
    :return:
    '''
    return np.sqrt(sum(np.power(v - u, 2)))


def adj(nodes):
    '''
    produce symmetric adjacency matrix for nodes
    :param nodes:
    :return:
    '''
    dim = len(nodes)
    M = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim):
        for j in range(i, dim):
            v = nodes[i]
            u = nodes[j]
            d = dist(v, u)
            M[i, j] = d

    return M + M.T


def crossover(p1, p2):
    '''
    PMX(partially mapped crossover)
    :param p1: permutation
    :param p2:
    :return:c1,c 2
    '''
    chromo_len = len(p1)
    loc1 = randint(0, chromo_len - 1)
    loc2 = randint(0, chromo_len - 1)
    if loc1 > loc2:
        loc2, loc1 = loc1, loc2
    c1 = np.zeros(shape=(chromo_len,), dtype=np.int32)
    c2 = np.zeros(shape=(chromo_len,), dtype=np.int32)
    # exchange subroute
    c1[loc1:loc2] = p2[loc1:loc2]
    c2[loc1:loc2] = p1[loc1:loc2]

    #     generate partial maps between the subroutes
    map1 = {k: v for k, v in zip(c1[loc1:loc2], c2[loc1:loc2])}
    map2 = {k: v for k, v in zip(c2[loc1:loc2], c1[loc1:loc2])}

    # first half for c1
    for i in range(loc1):
        v = p1[i]
        while v in map1:
            v = map1[v]
        c1[i] = v

    # first half for c2
    for i in range(loc1):
        v = p2[i]
        while v in map2:
            v = map2[v]
        c2[i] = v

    # second half for c1
    for i in range(loc2, chromo_len):
        v = p1[i]
        while v in map1:
            v = map1[v]
        c1[i] = v

    # second half for c2
    for i in range(loc2, chromo_len):
        v = p2[i]
        while v in map2:
            v = map2[v]
        c2[i] = v

    return c1, c2


def mutate(c, p_m):
    '''
    swap mutation
    :param c:
    :param p_m:
    :return:
    '''
    # for _ in range(int(len(c)/2)):
    if random() < p_m:
        i = randint(0, len(c) - 1)
        j = randint(0, len(c) - 1)
        if i != j:
            c[i], c[j] = c[j], c[i]


class TSPGA:
    def __init__(self, pop_size=10, p_c=0.5, p_m=0.5, max_iter=100, record_intermediate=False, debug=True, selection_method='rws', tournament_size=2, greedy_intialize_population=True):
        self.pop_size = pop_size
        self.code_len = None
        self.M = None
        self.p_c = p_c
        self.p_m = p_m
        self.max_iter = max_iter
        self.record_intermediate = record_intermediate
        self.debug = debug
        self.epoch_info = {'std': [], 'avg': []}
        self.k = tournament_size
        self.greedy_initialize_population = greedy_intialize_population

        if selection_method == 'rws':
            self.select = self.wheel_select
        elif selection_method == 'tournament':
            self.select = self.tournament_select
        else:
            raise Exception("unknown method")


    def init_population(self):
        '''
        first generate greedy solution to ensure solution space contains some good result, then fill the rest with random permutation
        :return:
        '''
        def greedy_solution(curr):
            '''
            nearest neighbour construction, solution starts with node `current`
            :return:
            '''
            sol = [0 for _ in range(self.code_len)]
            sol[0] = curr
            visited = [False] * self.code_len
            visited[curr] = True
            for pos in range(1, self.code_len):
                min = 9999999
                for j in range(self.code_len):
                    if not visited[j] and self.M[curr, j] < min:
                        min = self.M[curr, j]
                        sol[pos] = j

                curr = sol[pos]
                visited[curr] = True

            return sol

        pop = []
        if self.greedy_initialize_population:
            # pop.append(greedy_solution(0))
            rounds = min(int(self.pop_size * 0.1), self.code_len, self.pop_size)
            for i in range(rounds):
                pop.append(greedy_solution(i))

        # generate random permutation for the rest
        for _ in range(len(pop), self.pop_size):
            pop.append(np.random.permutation(self.code_len))

        # for i in range(self.pop_size):
        #     chromo = np.random.permutation(self.code_len)
        #     pop.append(chromo)
        return pop

    def init_params(self, problem):
        '''
        init the adjacency matrix
        :param problem: shape (len(nodes, 3)
        :return: void
        '''
        problem = np.array(problem)
        self.code_len = len(problem)  # node permutation encoding
        nodes = problem[:, 1:]
        self.M = adj(nodes)

    def get_fitness(self, D):
        '''
        transform distance into fitness
        :param D: distance
        :return: fitness
        '''

        max_d = np.max(D)
        f = max_d - D
        return f

    def get_distance(self, pop):
        D = []
        for ind in pop:
            d = 0
            for i in range(len(ind) - 1):
                d += self.M[ind[i], ind[i + 1]]

            d += self.M[ind[-1], ind[0]]
            D.append(d)

        return np.array(D)

    def find_idx(self, r, arr):
        '''
        :param r: random number
        :param arr: the cumulative prob
        :return: the sampling index
        '''
        # TODO: change to O(log(n))
        i = 0
        for i, v in enumerate(arr):
            if r < v:
                break
        return i

    def select(self, pop, f):
        pass

    def wheel_select(self, pop, f):
        '''
        RWS according to f
        :param pop: the population
        :param u: utility for each item
        :return: the mating pool, size equal to population
        '''
        res = []
        p_s = softmax(f)
        c_s = np.cumsum(p_s)
        for _ in range(len(pop)):
            r = random()
            res.append(pop[self.find_idx(r, c_s)])
        return np.array(res)

    def tournament_select(self, pop, f):
        '''
        unbiased tournament selection
        :param pop: permutation code representing node in the path of code_len
        :param f: fitness for each permutation code
        :param k: tournament size
        :return: mating pool
        '''
        k = self.k
        def get_winner(tournament):
            '''
            get winner from tournament
            :param tournament:
            :return: winner path
            '''
            t_id = np.argmax([f[c_id] for c_id in tournament])
            w_id = tournament[t_id]
            return pop[w_id]


        tournaments = np.zeros((k, self.pop_size), dtype=np.int32)
        pop_mate = np.zeros((self.pop_size, self.code_len), dtype=np.int32)
        for i in range(k):
            tournaments[i] = np.random.permutation(self.pop_size)

        for i in range(self.pop_size):
            tournament = tournaments[:, i]
            pop_mate[i] = get_winner(tournament)
        return pop_mate


    def solve(self, problem):

        self.init_params(problem)  # init params like adj matrix, code_len

        # 1. generate population
        pop = self.init_population()
        pop_size = len(pop)
        if pop_size % 2 == 1:
            pop_size -= 1
            pop = pop[:-1]

        d = self.get_distance(pop)
        avg = np.average(d)
        idx = np.argmin(d)
        sol = np.copy(pop[idx])

        round_found_optimal = -1

        f = self.get_fitness(d)

        min_dist = d[idx]

        if self.debug:
            print(f"average distance before: {avg}")
        if self.record_intermediate:
            self.epoch_info['avg'].append(avg)
            self.epoch_info['std'].append(np.std(d))

        iter = 0
        while iter < self.max_iter:
            # 2. selection for next population
            # pop_mate = self.wheel_select(pop, f)
            pop_mate = self.select(pop, f)

            pop = []
            # choices = np.random.choice(np.arange(0, self.code_len), replace=False)
            choices = np.random.permutation(self.pop_size)
            for i in np.arange(0, pop_size, 2):
                p1 = pop_mate[choices[i]]
                p2 = pop_mate[choices[i + 1]]
                # 3.crossover
                if random() < self.p_c:
                    c1, c2 = crossover(p1, p2)
                else:
                    c1, c2 = p1, p2
                # 4. mutation
                for c in [c1, c2]:
                    mutate(c, self.p_m)

                    pop.append(c)

            d = self.get_distance(pop)
            f = self.get_fitness(d)
            idx = np.argmin(d)

            if d[idx] < min_dist:
                min_dist = d[idx]
                sol = np.copy(pop[idx])
                round_found_optimal = iter

            if self.debug and not self.record_intermediate:
                avg = np.mean(d)
                print(f"average distance in {iter}th round: {avg}")
            elif self.record_intermediate:
                avg = np.mean(d)
                self.epoch_info['std'].append(np.std(d))
                self.epoch_info['avg'].append(avg)

            # if similarity(pop) > 0.9:
            #     print(f"similarity > 0.9, break in {iter}th round")
            #     break

            iter += 1
        return min_dist, sol, round_found_optimal


def get_min_distance(M, tour):
    '''
    get minimum distance as given by the tour
    :param M: the adj matrix
    :param tour: the tour is a permutation of nodes
    :return: minimum distance
    '''
    d = 0
    for i in range(len(tour) - 1):
        u = tour[i] - 1  # -1 since tour node is start indexed by 1
        v = tour[i + 1] - 1
        d += M[u, v]
    d += M[tour[-1] - 1, tour[0] - 1]
    return d

def get_problem2():
    locations = or_example.locations
    problems = np.zeros((len(locations), 3), dtype=np.int32)
    for i, (x,y) in enumerate(locations):
        problems[i] = [i, x, y]

    return problems

if __name__ == "__main__":

    # # file_name = 'a280'
    # file_name = 'bayg29'
    # data_dir = 'data/tsp'
    # # solution_name = 'a280_optimal'
    # solution_name = 'bayg29_optimal'
    # tour = read_tsp_solution(os.path.join(data_dir, solution_name))
    #
    # problem = read_problem_tsp(os.path.join(data_dir, file_name))
    # problem = np.array(problem)

    problem = get_problem2()


    # print(len(problem))
    # ga = TSPGA(pop_size=5000, max_iter=100, p_c=0.5, p_m=0.1, greedy_intialize_population=True, selection_method='tournament', tournament_size=2)
    ga = TSPGA(pop_size=100, max_iter=10000, p_c=0.5, p_m=0.5, greedy_intialize_population=False, selection_method='tournament', tournament_size=2,debug=False)
    min_dist, sol, round_found = ga.solve(problem)


    # print(f"minimum distance of tour according to solution: {get_min_distance(ga.M, tour)}")

    print(f"min dist: {min_dist}")
    print(f"found tour (start index 1): {sol + 1}")
    print(f"found at round: {round_found}")
