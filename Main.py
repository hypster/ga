from random import random
from random import randint
import numpy as np

from DP import DP


if __name__ == '__main__':
    gen_len = 10
    w = np.array([randint(1, 10) for _ in range(gen_len)])
    print(f"weight: {w}")
    u = np.array([i for i in range(1, gen_len + 1)])
    max_w = 40

    v, idx = DP.solve(w, u, max_w)
    print(f"solution idx: {idx}")
    print(f"solution weight: {sum(w[i] for i in idx)}")


