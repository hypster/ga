import numpy as np
import sys

from util import read_problem_knap


class DP:
    @staticmethod
    def solve(W, V, MAX_W):
        # DA tableau to solve 0-1 knapsack program
        tbl = np.zeros((len(V)+1, MAX_W+1))
        keep = np.zeros_like(tbl)
        V = np.concatenate([[0], V])
        W = np.concatenate([[0], W])
        for i in range(1, len(V)):
            for w in range(MAX_W+1):
                if w - W[i] >= 0 and tbl[i-1,w] < V[i] + tbl[i-1, w - W[i]]:
                        keep[i,w] = 1
                        tbl[i,w] = V[i] + tbl[i-1, w - W[i]]
                else:
                    tbl[i,w] = tbl[i-1, w]

        # print(tbl)
        print(f"Optimized value: {tbl[-1,-1]}")

        k = MAX_W
        idx = []
        for i in range(len(V)-1, 0, -1):
            if keep[i, k] == 1:
                idx.append(i-1)
                k -= W[i]
        idx = sorted(idx)

        return tbl[-1,-1], idx




if __name__ == "__main__":
    # V = [10, 40, 30, 50]
    # W = [5, 4, 6, 3]
    # MAX_W = 10
    # v, idx = DP.solve(W, V, MAX_W)
    w,v,max_w, sol = read_problem_knap('/Users/septem/Documents/code/GA/data/instances_01_KP/large_scale/knapPI_1_100_1000_1')
    w = np.array(w)
    v = np.array(v)
    optimal_v,idx = DP.solve(w,v,max_w)
    print(f"idx found by solution: {idx}")

    print(f"solution given by the file")
    print(np.nonzero(sol))
    print(f"total weight: {sum(sol * w)}")
    print(f"total utility: {sum(sol * v)}")
