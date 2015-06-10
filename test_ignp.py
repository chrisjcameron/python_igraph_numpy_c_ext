import igraph as ig
import numpy as np
import ignp_fun

g = ig.Graph.Tree(127,2)

# Test Degree Summation
test_sum = bool(sum(g.degree())==ignp_fun.sum_degree(g))
print("Python sum == C sum: {}".format(test_sum))

# Test Degree Getters
dl = np.zeros(shape=len(g.degree()), dtype='int32')
ignp_fun.degree_array(g, dl)
test_get_degree = np.array_equal(dl, g.degree())
print("Python degree list == C degree list: {}".format(test_get_degree))
