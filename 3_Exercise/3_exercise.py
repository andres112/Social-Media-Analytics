import networkx as nx
import time
from kclique import k_clique_communities
import matplotlib.pyplot as plt
from girvan_newman import girvan_newman


def drawRuntime(data_cpm, data_gn):
    lists = sorted(data_cpm.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    # CPM plot
    plt.subplot(211)
    plt.plot(x, y)
    plt.title('CPM')
    plt.ylabel("Time (s)")
    plt.grid(True)

    lists = sorted(data_gn.items())
    x, y = zip(*lists)   
    # Girvan - Newman Plot
    plt.subplot(212)
    plt.plot(x, y, 'r*-')
    plt.title('Girvan Newman')
    plt.ylabel("Time (s)")
    plt.xlabel("Number of Nodes")
    plt.grid(True)

    plt.suptitle('Runtime performance')
    plt.show()

def girvanNewman(graph):
    communities = girvan_newman(graph)
    return tuple(sorted(c) for c in next(communities))

if __name__ == '__main__':
    G = nx.read_edgelist('DataSets/email-Eu.txt')  # Undirected Graph
    nodes = nx.nodes(G)

    # CPM part
    # get initial edges to form the graph for the CPM task
    cpm_times = {}
    print("*** Performance for Clique Percolation Method ***")
    for i in range(50000, nx.number_of_nodes(G), 1000):
        cpm_graph = G.subgraph(list(nodes)[:i])
        print("---\nNodes: ", nx.number_of_nodes(cpm_graph))
        print("Edges: ", nx.number_of_edges(cpm_graph))
        start = time.time()
        c = k_clique_communities(cpm_graph, 4)
        end = time.time()
        print("Time in seconds: ",end - start)
        cpm_times[i] = end - start

    # Girman Newman part
    # get initial edges to form the graph for the GN task
    gn_times = {}
    print("*** Performance for Girman Newman Algortihm ***")
    for i in range(100, 1001, 100):
        gn_graph = G.subgraph(list(nodes)[:i])
        print("---\nNodes: ", nx.number_of_nodes(gn_graph))
        print("Edges: ", nx.number_of_edges(gn_graph))
        start = time.time()
        c = girvanNewman(gn_graph)
        end = time.time()
        print("Time in seconds: ",end - start)
        gn_times[i] = end - start
    
    # plot the results for both algorithms
    drawRuntime(cpm_times, gn_times)