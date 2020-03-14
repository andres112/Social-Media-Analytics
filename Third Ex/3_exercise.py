import networkx as nx
import time
from kclique import k_clique_communities
import matplotlib.pyplot as plt
from girvan_newman import girvan_newman


def drawGraph(graph):
    # Drawing the Graph using matplotlib
    nx.draw_networkx(graph, node_color=['red'], with_labels=True)
    plt.show()

def girvanNewman(graph):
    communities = girvan_newman(graph)
    return tuple(sorted(c) for c in next(communities))

if __name__ == '__main__':
    G = nx.read_edgelist('DataSets/email-Eu.txt')  # Undirected Graph
    nodes = nx.nodes(G)

    # CPM part
    # get initial edges to form the graph for the CPM task
    cpm_times = []
    for i in range(50000, nx.number_of_nodes(G), 1000):
        cpm_graph = G.subgraph(list(nodes)[:i])
        # print("Nodes: ", nx.number_of_nodes(cpm_graph))
        # print("Edges: ", nx.number_of_edges(cpm_graph))
        start = time.time()
        c = k_clique_communities(cpm_graph, 4)
        end = time.time()
        # print(end - start)
        cpm_times.append({i:end - start})
    
    print(cpm_times)

    # Girman Newman part
    # get initial edges to form the graph for the GN task
    gn_times = []
    for i in range(100, 1000, 100):
        gn_graph = G.subgraph(list(nodes)[:i])
        # print("Nodes: ", nx.number_of_nodes(gn_graph))
        # print("Edges: ", nx.number_of_edges(gn_graph))
        start = time.time()
        c = girvanNewman(gn_graph)
        end = time.time()
        # print(end - start)
        gn_times.append({i:end - start})

        # print("*** Number of communities after 1st iteration Givar-Newman: {}\n".format(
        #     len(girvanNewman(gn_graph))))

    print("************************************\n",gn_times)