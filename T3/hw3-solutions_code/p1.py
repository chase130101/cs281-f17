"""
Solution to HW3, Problem 1
"""

import torch 
from torch.autograd import Variable
import itertools
import numpy as np
import pdb

def construct_graph():
    people = ["SR", "YD", "MG", "ZH", "HS", "RS", "NZ", "YK"]
    edges = [(2, 5), (1, 5), (3, 5), (3, 4), (3, 6), (3, 7)]
    friends_matrix = torch.zeros((8, 8))
    for u, v in edges:
        friends_matrix[u][v] = 2.
        friends_matrix[v][u] = 2.
    return friends_matrix

def new_construct_graph():
    edges = [(2, 5), (1, 5), (3, 5), (3, 4), (3, 6), (3, 7)]
    potentials = [2, -2, -2, -8, -2, 3, -2, 1]

    # make a tensor with [edge potentials ... unary potentials]
    as_tensor = torch.zeros(len(edges) + len(potentials))
    as_tensor[:len(edges)] = 2
    as_tensor[len(edges):] = torch.Tensor(potentials)

    graph_var = Variable(as_tensor, requires_grad=True)
    return edges, graph_var

def score(a, graph_desc, graph_var, i=0):
    """
    a:          assignment vector (torch.Tensor)
    graph_desc: edges (as pairs (i, j))
    graph_var : Variable containing edge potentials, and unary_potentials
    """
    score = Variable(torch.zeros(1))
    for i, (u, v) in enumerate(graph_desc):
        score += (graph_var[i] * a[u] * a[v])
    for i in range(len(a)):
        score += (graph_var[len(graph_desc) + i] * a[i])
    return score

def brute_force_log_partition(graph_desc, graph_var):
    assignments = [torch.FloatTensor(seq) for seq in itertools.product([0, 1], repeat=8)]
    scores = torch.cat([score(assignment, graph_desc, graph_var) for assignment in assignments], 0)
    return scores.exp().sum(0, True).log()

def brute_force_marginalization(graph_desc, graph_var, person):
    marginal = filter(lambda seq: seq[person] == 1, [list(seq) for seq in itertools.product([0, 1], repeat=8)])
    tensors = map(lambda l: torch.Tensor(l), marginal)
    scores = torch.cat([score(assignment, graph_desc, graph_var) for assignment in tensors])
    return scores.exp().sum(0, True).log()

def brute_force_log_probability(graph_desc, graph_var, person, Z):
    score = brute_force_marginalization(graph_desc, graph_var, person)
    return np.exp(score.data[0] - Z.data[0])

def dfs(i, parent, beliefs, messages, edges, unary_potentials):
    # visit all the children to find their beliefs
    for u, v in edges:
        if u == i and v != parent:
            dfs(v, i, beliefs, messages, edges, unary_potentials)

    # calculate belief 
    beliefs[i][0] = messages[:, i, 0].sum()
    beliefs[i][1] = messages[:, i, 1].sum() + unary_potentials[i]
    beliefs[i] -= np.log(np.sum(np.exp(beliefs[i])))

    # calculate message - note that since our edges are 0 when either is 0, it's not very complicated
    if parent is not None:
        messages[i][parent][1] = 2 + beliefs[i][1]

def serial_mp(i, graph_desc, unary_potentials):
    # compute the log-marginal probability for a node i 
    messages = np.zeros((8, 8, 2))
    beliefs = np.zeros((8, 2))
    edges = graph_desc + [(v, u) for u, v in graph_desc]
    dfs(i, None, beliefs, messages, edges, unary_potentials)
    print(np.exp(beliefs[i]))

def autograd_marginal_probabilities(graph_desc, graph_var):
    loss = brute_force_log_partition(graph_desc, graph_var)
    loss.backward()
    grads = graph_var.grad[len(graph_desc):].data
    return grads

def main():
    graph_desc, graph_var = new_construct_graph()

    # (a)
    a = score(torch.Tensor([1, 0, 0, 0, 0, 1, 0, 0]), graph_desc, graph_var)[0]
    print(a.data[0])

    # (b)
    Z = brute_force_log_partition(graph_desc, graph_var)
    print(Z.data[0])
    print(np.exp(a.data[0] - Z.data[0]))

    # (c)
    RS = brute_force_marginalization(graph_desc, graph_var, 5)
    print(RS.data[0])
    print(brute_force_log_probability(graph_desc, graph_var, 5, Z))

    # (e)
    autograd_marginals = autograd_marginal_probabilities(graph_desc, graph_var)
    for i in range(8):
        print(brute_force_log_probability(graph_desc, graph_var, i, Z), autograd_marginals[i])

    # (f)
    print(serial_mp(0, graph_desc, graph_var[len(graph_desc):].data.numpy()))

if __name__ == '__main__':
    main()