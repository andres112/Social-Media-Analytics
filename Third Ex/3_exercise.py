import networkx as nx
from networkx.algorithms.community import k_clique_communities
import matplotlib.pyplot as plt

if __name__ == '__main__':
    G = nx.read_edgelist('DataSets/email-Eu.txt') # Undirected Graph
    print (nx.info(G))

    c = list(k_clique_communities(G, 4))

    h = sorted(list(c[0]))
    print(len(h))

    # Drawing the Graph using matplotlib
    # nx.draw_networkx(UG, node_color=['red'], with_labels=True)
    # plt.show()