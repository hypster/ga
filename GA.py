import os

import numpy as np
from random import random
from random import randint

from comp_func import softmax
from util import read_problem_knap


def get_fitness(pop, u):
    '''
    :param pop: population
    :param u: utility for each item
    :return: result array of utility value for each individual
    '''
    res = []
    for ind in pop:
        f = 0
        for i, v in enumerate(ind):
            f += v * u[i]
        res.append(f)
    return np.array(res)


def find_idx(r, arr):
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


def wheel_select(pop, u):
    '''
    :param pop: the population
    :param u: utility for each item
    :return: the mating pool, size equal to population
    '''
    res = []
    f = get_fitness(pop, u)
    p_s = softmax(f)
    c_s = np.cumsum(p_s)
    for _ in range(len(pop)):
        r = random()
        res.append(pop[find_idx(r, c_s)])
    return np.array(res)


def crossover(p1, p2):
    '''single point crossover
    :param p1:
    :param p2:
    :return: c1,c2 children
    '''
    p = randint(0, len(p1) - 1)
    c1 = np.concatenate([p1[:p], p2[p:]])
    c2 = np.concatenate([p2[:p], p1[p:]])
    return c1, c2


def mutate(c, p):
    '''
    :param c: individual string
    :param p: prob for mutation
    :return: void
    '''
    for i, g in enumerate(c):
        if random() < p:
            c[i] = 1 - g


def similarity(pop):
    ''' calculate similarity between population, using the average value of similarity on each position
    :param pop: population
    :return: average similarity
    '''
    pop = np.array(pop)
    n = len(pop)
    res = []
    for j in range(len(pop[0])):
        res.append((n - sum(pop[:, j])) / n)

    return np.average(np.array(res))


class GA:
    def __init__(self, pop_size=100, p_c=0.5, p_m=0.1, max_iter=100, repair_scheme="greedy", record_intermediate=False, debug = True):
        '''
        init params
        :param pop_size: population size
        :param p_c: prob for crossover
        :param p_m: prob for mutation
        :param max_iter: max iteration
        :param repair_scheme: {"greedy", "random"} default: greedy
        :return:
        '''
        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        self.max_iter = max_iter
        self.record_intermediate = record_intermediate
        self.debug = debug
        self.code_len = None
        self.w = None
        self.max_w = None
        self.u = None
        self.profit_weight_ratio_sorted = None
        self.epoch_info = {'std': [], 'avg': []}
        if repair_scheme == 'greedy':
            self.repair = self.repair_greedy
        else:
            self.repair = self.repair_random

    def initialize_pop(self):
        '''
        :param ratio_sort_idx: sorted index according to profit to weight ratio
        :param pop_size: size of the population
        :param length: length of the individual string
        :param w: weight for each item
        :param max_w: maximum allowed total weight
        :return: random population
        '''

        def create_individual(p=0.5):
            '''
            :param chrom_len: length of the individual string
            :param p: flip rate for binary encoding
            :return: random individual string
            '''
            arr = []
            for _ in range(self.code_len):
                r_bit = int(random() < p)
                arr.append(r_bit)
            return np.array(arr)


        res = []
        for i in range(self.pop_size):
            ind = create_individual()
            if self.get_weight(ind, self.w) > self.max_w:
                self.repair(ind)
            res.append(ind)
        return np.array(res)

    def repair(self, ind):
        pass

    def _init_param(self, w, u, max_w):
        self.w = w
        self.u = u
        self.max_w = max_w
        self.code_len = len(w)
        ratio = u / w
        _temp = sorted([(i, r) for i, r in enumerate(ratio)], key=lambda x: x[1])
        self.profit_weight_ratio_index_sorted = [i for i, v in _temp]

    def get_weight(self, ind, w):
        return np.sum(ind * w)

    def repair_random(self, ind):
        '''
        random repair
        :param ind:
        :return:
        '''
        while self.get_weight(ind) > self.max_w:
            p = randint(0, len(ind) - 1)
            while ind[p] == 0:
                p = randint(0, len(ind) - 1)
            ind[p] = 0

    def repair_greedy(self, ind):
        '''
        Greedy repair according the profit to weight ratio
        :param ind:
        :return:
        '''
        curr_w = self.get_weight(ind, self.w)
        excess = curr_w - self.max_w
        repaired = False
        while not repaired:
            for i in self.profit_weight_ratio_index_sorted:
                if ind[i] == 1:
                    ind[i] = 0
                    excess = excess - self.w[i]
                    if excess < 0:
                        repaired = True
                        break
            if repaired:
                break

    def solve(self, w, u, max_w):
        '''
        :param w: weight for each item
        :param u: the utility for each item
        :param max_w: max weight allowed
        :param pop_size: population size, hyperparameter, default 100
        :param p_c: prob for crossover
        :param p_m: prob for mutation
        :return: the optimal utility and the idx
        '''
        self._init_param(w, u, max_w)
        sol = None
        round_found_optimal = 0
        max_v = -100

        # 1. initialize random population
        pop = self.initialize_pop()
        # print("population")
        # print(pop)
        f = get_fitness(pop, u)
        avg = np.mean(f)
        idx = np.argmax(f)

        if f[idx] > max_v:
            max_v = f[idx]
            sol = np.copy(pop[idx])
        if self.debug:
            print(f"average fitness before: {avg}")
        if self.record_intermediate:
            self.epoch_info['avg'].append(avg)
            self.epoch_info['std'].append(np.std(f))

        iter = 0
        while iter < self.max_iter:
            # 2. selection for next population
            pop_mate = wheel_select(pop, u)

            pop = []
            choices = np.random.choice(np.arange(0, self.pop_size), size=self.pop_size, replace=False)
            for i in np.arange(0, len(choices), 2):
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

                    self.repair(c)
                    pop.append(c)

            f = get_fitness(pop, u)
            idx = np.argmax(f)
            if f[idx] > max_v:
                max_v = f[idx]
                sol = np.copy(pop[idx])
                round_found_optimal = iter

            if self.debug and not self.record_intermediate:
                avg = np.mean(f)
                print(f"average fitness in {iter}th round: {avg}")
            elif self.record_intermediate:
                avg = np.mean(f)
                self.epoch_info['std'].append(np.std(f))
                self.epoch_info['avg'].append(avg)


            # if similarity(pop) > 0.9:
            #     print(f"similarity > 0.9, break in {iter}th round")
            #     break

            iter += 1
        return max_v, sol, round_found_optimal


if __name__ == "__main__":
    file_name = 'knapPI_1_5000_1000_1'
    # file_name = "knapPI_1_200_1000_1"
    data_dir = 'data/instances_01_KP/large_scale'
    w, u, max_w, sol = read_problem_knap(os.path.join(data_dir, file_name))

    w = np.array(w)
    u = np.array(u)
    max_v, ga_sol, round = GA().solve(w, u, max_w)

    print(f"solution found by GA:\noptimal value: {max_v}\tweight: {sum(ga_sol*w)} \tround: {round}")
    print(f"include items: {np.nonzero(ga_sol)}")

    print(f"solution given by the file")
    print(np.nonzero(sol))
    print(f"total weight: {sum(sol * w)}")
    print(f"total utility: {sum(sol * u)}")
