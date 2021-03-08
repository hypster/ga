from unittest import TestCase

import numpy as np

from GA import GA
from util import read_problem_knap, read_tsp_solution
import os
from TSPGA import adj

class Test(TestCase):
    def test1(self):
        data_dir = 'data/instances_01_KP/large_scale'
        for p in os.listdir(data_dir):
            print(p)
            w,u,max_w, sol = problem = read_problem_knap(os.path.join(data_dir, p))
            w = np.array(w)
            u = np.array(u)
            max_v, ga_sol, round = GA().solve(w, u, max_w)

            print(f"solution found by GA:\noptimal value: {max_v}\tweight: {sum(ga_sol * w)} \tround: {round}")
            print(f"include items: {np.nonzero(ga_sol)}")

            print(f"solution given by the file")
            print(np.nonzero(sol))
            print(f"total weight: {sum(sol * w)}")
            print(f"total utility: {sum(sol * u)}")

    def test2(self):
        X = np.array([[1,2],
             [3,4],
             [5,6]])
        print(adj(X))


    def test3(self):
        p1 = [1,2,3,4,5,6,7,8,9]
        p2 = [5,4,6,9,2,1,7,8,3]

        loc1 = 2
        loc2 = 6
        chromo_len = 9
        c1 = np.zeros(shape=(chromo_len,))
        c2 = np.zeros(shape=(chromo_len,))
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

        print(c1, c2)

    def test4(self):
        tour = read_tsp_solution('data/tsp/a280_optimal')
        print(len(tour))








