import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K = range(1, 5)
seeds = range(6)
seed_min_idx = []
min_costs = []
for k in K:
    costs = []
    for seed in seeds:
        mixture, post = common.init(X, k, seed)
        mixture_updated, post_updated, cost = kmeans.run(X, mixture, post)
        costs.append(cost)
    min_costs.append(min(costs))
    seed_min_idx.append(costs.index(min(costs)))

# plotting
# i = 0
# for k in K:
#     mixture, post = common.init(X, k, seed_min_idx[i])
#     mixture_updated, post_updated, cost = kmeans.run(X, mixture, post)
#     common.plot(X, mixture_updated, post_updated, "K = " + str(k))
#     i += 1

# print(min_costs)
# print(seed_min_idx)
