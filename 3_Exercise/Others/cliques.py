from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt


def k_cliques(graph):
    # 2-cliques
    cliques = [{i, j} for i, j in graph.edges() if i != j]
    k = 2

    while cliques:
        # result
        yield k, cliques

        # merge k-cliques into (k+1)-cliques
        cliques_1 = set()
        for u, v in combinations(cliques, 2):
            w = u ^ v
            if len(w) == 2 and graph.has_edge(*w):
                cliques_1.add(tuple(u | w))

        # remove duplicates
        cliques = list(map(set, cliques_1))
        k += 1


def print_cliques(graph, size_k):
    for k, cliques in k_cliques(graph):
        if k == size_k:
            print('%d-cliques = %d, %s.' % (k, len(cliques), cliques))


size_k = 4
graph = nx.read_edgelist('../DataSets/got.txt')
print(nx.info(graph))
nx.draw_networkx(graph, node_color=['red'], with_labels=True)
plt.show()
print_cliques(graph, size_k)