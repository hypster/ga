def read_problem_knap(filepath):
    '''
    read dataset, the first line is n, max_w
    the following n lines are v_i, w_i
    the last line gives the solution in terms of the 0-1 array
    :param filepath:
    :return:
    '''
    w,v = [], []
    n,max_w = None, None
    with open(filepath) as f:
        n,max_w = f.readline().split(" ")
        n = int(n)
        max_w = int(max_w)
        for _ in range(n):
            line = f.readline()
            a,b = line.split(" ")
            # print(a,b)
            # break
            a = int(a)
            v.append(a)
            b = int(b)
            w.append(b)
        sol = [int(i) for i in f.readline().split(' ')]

    return w,v,max_w, sol

def read_problem_tsp(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        dim = int(lines[3].split(" ")[-1])
        problem = []
        for line in lines[6:6+dim]:
            problem.append([float(item) for item in line.split()])

        return problem

def read_tsp_solution(filepath, start_line = 4):
    tour = []
    with open(filepath) as f:
        for line in f.readlines()[start_line:]:
            tour.append(int(line))
        return tour[:-1]


